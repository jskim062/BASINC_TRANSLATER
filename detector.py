import os
import cv2
import numpy as np
import onnxruntime as ort

class ComicTextDetector:
    def __init__(self, model_path="comictextdetector.pt.onnx"):
        self.model_path = model_path
        # Use CPU execution provider as default
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.target_size = 1024

    def detect(self, image_path, conf_threshold=0.4, nms_threshold=0.4):
        """
        Detects text in the comic image and returns bounding boxes.
        Returns format: list of dicts with 'box_2d': [ymin, xmin, ymax, xmax] (0-1000 normalized)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        h, w = img.shape[:2]

        # 1. Preprocess (Letterbox scaling to target size)
        scale = min(self.target_size / h, self.target_size / w)
        nh, nw = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (nw, nh))

        pad_w = self.target_size - nw
        pad_h = self.target_size - nh
        
        # padding strategy: you can pad bottom/right or center. yolov5 usually pads center or bottom/right.
        # We'll use bottom/right padding
        img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Channel ordering: BGR to RGB, HWC to CHW
        input_tensor = img_padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        input_tensor /= 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # 2. Inference
        blk, _, _ = self.session.run(["blk", "seg", "det"], {"images": input_tensor})
        preds = blk[0] # Shape: (64512, 7)
        
        # 3. Post-process (Filter by confidence and apply NMS)
        boxes = []
        scores = []
        
        # preds: [cx, cy, w, h, obj_conf, cls0_conf, cls1_conf]
        for row in preds:
            obj_conf = row[4]
            # Combine obj_conf with class confidence? Assuming column 6 is text.
            class_conf = max(row[5], row[6])
            score = obj_conf * class_conf
            
            if score >= conf_threshold:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                
                # YOLO format cx, cy, w, h to xmin, ymin, w, h for NMS
                xmin = cx - bw / 2.0
                ymin = cy - bh / 2.0
                
                boxes.append([int(xmin), int(ymin), int(bw), int(bh)])
                scores.append(float(score))

        # Perform Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, scores, float(conf_threshold), float(nms_threshold))
        
        detected_items = []
        final_boxes = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                bx, by, bw, bh = boxes[i]
                
                # Convert back to unpadded, unscaled image coordinates
                x_min = bx / scale
                y_min = by / scale
                x_max = (bx + bw) / scale
                y_max = (by + bh) / scale
                
                # Clip to image boundaries
                x_min = max(0, min(x_min, w))
                y_min = max(0, min(y_min, h))
                x_max = max(0, min(x_max, w))
                y_max = max(0, min(y_max, h))
                
                final_boxes.append([x_min, y_min, x_max, y_max])
                
        # Save raw line boxes for accurate masking before merge
        raw_line_boxes = []
        for box in final_boxes:
            n_xmin = int(box[0] / w * 1000)
            n_ymin = int(box[1] / h * 1000)
            n_xmax = int(box[2] / w * 1000)
            n_ymax = int(box[3] / h * 1000)
            raw_line_boxes.append([n_ymin, n_xmin, n_ymax, n_xmax])
            
        # Merge nearby boxes into text blocks
        merge_margin = min(w, h) * 0.005  # 0.5% of image size margin
        merged_boxes = self._merge_boxes(final_boxes, merge_margin)
        
        for mb in merged_boxes:
            # Normalize to 0-1000 (Gemini format)
            n_xmin = int(mb[0] / w * 1000)
            n_ymin = int(mb[1] / h * 1000)
            n_xmax = int(mb[2] / w * 1000)
            n_ymax = int(mb[3] / h * 1000)
            
            detected_items.append({
                "box_2d": [n_ymin, n_xmin, n_ymax, n_xmax],
                # Style and placeholder text so translator logic is somewhat compatible
                "style": "standard",
                "translated": "", # Will be filled by Gemini
                "original": "",
                "font_size": None
            })
                
        return detected_items, raw_line_boxes

    def _merge_boxes(self, boxes, merge_margin):
        if not boxes: 
            return []
            
        n = len(boxes)
        parent = list(range(n))
        
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                
        for i in range(n):
            for j in range(i + 1, n):
                b1 = boxes[i]
                b2 = boxes[j]
                
                # Expand box 1 by margin
                b1_xmin = b1[0] - merge_margin
                b1_ymin = b1[1] - merge_margin
                b1_xmax = b1[2] + merge_margin
                b1_ymax = b1[3] + merge_margin
                
                # Check for intersection
                if not (b1_xmax < b2[0] or b1_xmin > b2[2] or 
                        b1_ymax < b2[1] or b1_ymin > b2[3]):
                    union(i, j)
                    
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = list(boxes[i])
            else:
                g = groups[root]
                b = boxes[i]
                g[0] = min(g[0], b[0])
                g[1] = min(g[1], b[1])
                g[2] = max(g[2], b[2])
                g[3] = max(g[3], b[3])
                
        return list(groups.values())
