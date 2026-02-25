from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import io
import base64
import cv2
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest

class ImageProcessor:
    def __init__(self, standard_font="NanumSquareRoundB.ttf", emphasis_font="NanumSquareRoundEB.ttf"):
        # Default Windows font directory if not relative/absolute path
        self.font_dir = "C:\\Windows\\Fonts\\"
        self.standard_font = self._resolve_font_path(standard_font)
        self.emphasis_font = self._resolve_font_path(emphasis_font)
        # Initialize LaMa inpainting model (CPU by default unless GPU is available and configured)
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.inpaint_model = ModelManager(name="lama", device=device)
        except Exception as e:
            print(f"Warning: Could not initialize LaMa model. Error: {e}")
            self.inpaint_model = None

    def _resolve_font_path(self, font_name):
        if os.path.isabs(font_name) or os.path.exists(font_name):
            return font_name
        
        # Check in Windows Fonts directory
        win_path = os.path.join(self.font_dir, font_name)
        if os.path.exists(win_path):
            return win_path
        
        # Check in local project directory
        local_path = os.path.join(os.getcwd(), font_name)
        if os.path.exists(local_path):
            return local_path
            
        return font_name # Fallback to name itself (PIL might still find it)

    def process_manga_page(self, image_path, translations, raw_boxes=None):
        """
        Overlays translated text onto the original manga page.
        Uses inpainting to remove original text if LaMa model is available.
        """
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        # Create a copy for the output
        output_img = img.copy()
        # --- Text Masking & Inpainting ---
        # If inpainter is loaded, we first try to remove the original text
        if self.inpaint_model:
            # Generate mask using raw line boxes if available, to preserve background art
            if raw_boxes:
                # Filter out raw boxes that belong to an SFX (empty translation)
                filtered_mask_source = []
                for raw_box in raw_boxes:
                    rymin, rxmin, rymax, rxmax = raw_box
                    r_area = max(0, rymax - rymin) * max(0, rxmax - rxmin)
                    
                    is_sfx = False
                    for t in translations:
                        t_text = t.get("formatted_text") or t.get("translated", "")
                        tymin, txmin, tymax, txmax = t.get("box_2d", [0,0,0,0])
                        
                        # Calculate intersection
                        iymin = max(rymin, tymin)
                        ixmin = max(rxmin, txmin)
                        iymax = min(rymax, tymax)
                        ixmax = min(rxmax, txmax)
                        
                        if ixmin < ixmax and iymin < iymax:
                            inter_area = (iymax - iymin) * (ixmax - ixmin)
                            if r_area > 0 and inter_area / r_area > 0.5: # 50% overlap threshold
                                if not t_text.strip():
                                    is_sfx = True
                                break
                    
                    if not is_sfx:
                        filtered_mask_source.append({"box_2d": raw_box})
                mask_source = filtered_mask_source
            else:
                mask_source = [t for t in translations if (t.get("formatted_text") or t.get("translated", "")).strip()]
                
            mask_np = self._create_inpaint_mask((width, height), mask_source)
            
            # 2. Convert original image to numpy array format expected by iopaint
            img_np = np.array(img.convert("RGB"))
            
            # 3. Perform inpainting
            try:
                # iopaint InpaintRequest expects base64 encoded strings
                _, buffer = cv2.imencode('.png', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                img_b64 = base64.b64encode(buffer).decode('utf-8')
                
                _, mask_buffer = cv2.imencode('.png', mask_np)
                mask_b64 = base64.b64encode(mask_buffer).decode('utf-8')
    
                config = InpaintRequest(
                    image="data:image/png;base64," + img_b64,
                    mask="data:image/png;base64," + mask_b64, 
                    ldm_steps=25,
                    ldm_sampler="plms",
                    hd_strategy="Original",
                    zits_wireframe=False,
                    cv2_flag="INPAINT_NS",
                    cv2_radius=4
                )
                
                if self.inpaint_model:
                    # iopaint의 모델 매니저는 보통 (image, mask, config) 순으로 받는데
                    # 만약 에러가 계속된다면 mask_np를 확실하게 uint8 타입으로 보장해줘.
                    mask_np = mask_np.astype(np.uint8)
                    inpainted_np = self.inpaint_model(img_np, mask_np, config)
                    
                    # 만약 결과가 memoryview라면 np.array로 다시 감싸기
                    if isinstance(inpainted_np, memoryview):
                        # memoryview를 bytes로 변환 후 numpy array로 재구성
                        inpainted_np = np.frombuffer(inpainted_np.tobytes(), dtype=np.uint8).reshape(img_np.shape)
                        
                    # Some models return BGR, we need RGB for PIL
                    if isinstance(inpainted_np, np.ndarray):
                        if len(inpainted_np.shape) == 3 and inpainted_np.shape[2] == 3:
                             # iopaint usually returns BGR
                             output_img = Image.fromarray(inpainted_np[:, :, ::-1].astype('uint8'))
                        else:
                             output_img = img.copy() # fallback
                    else:
                        print(f"Unexpected inpainting result type: {type(inpainted_np)}")
                        output_img = img.copy() # fallback
                        
                else:
                    output_img = img.copy()
            except Exception as e:
                import traceback
                print(f"Inpainting failed: {e}")
                traceback.print_exc()
                output_img = img.copy() # fallback
            
        draw = ImageDraw.Draw(output_img)
        
        # --- Text Overlay ---
        for item in translations:
            box = item.get("box_2d")
            # Try to get formatted_text first, fallback to standard translated
            translated_text = item.get("formatted_text") or item.get("translated")
            style = item.get("style", "standard")
            
            if not box or len(box) != 4 or not translated_text:
                continue
                
            ymin, xmin, ymax, xmax = box
            left = xmin * width / 1000
            top = ymin * height / 1000
            right = xmax * width / 1000
            bottom = ymax * height / 1000
            
            # --- Draw text ---
            # Font selection based on style
            target_font_path = self.emphasis_font if style == "emphasis" else self.standard_font
            
            self._draw_dynamic_wrapped_text(draw, translated_text, (left, top, right, bottom), target_font_path, fill=(0, 0, 0))
            
        return output_img

    def _create_inpaint_mask(self, image_size, translations):
        """Creates a binary numpy mask where text boxes are white (255) and background is black (0)."""
        width, height = image_size
        mask = Image.new('L', (width, height), 0) # Black background
        draw = ImageDraw.Draw(mask)
        
        for item in translations:
            box = item.get("box_2d")
            if not box or len(box) != 4:
                continue
                
            ymin, xmin, ymax, xmax = box
            # Expand mask slightly to cover edges of text/bubbles better
            padding = int(min(width, height) * 0.005) # 0.5% padding
            
            left = max(0, (xmin * width / 1000) - padding)
            top = max(0, (ymin * height / 1000) - padding)
            right = min(width, (xmax * width / 1000) + padding)
            bottom = min(height, (ymax * height / 1000) + padding)
            
            draw.rectangle([left, top, right, bottom], fill=255) # White text box
            
        return np.array(mask)

    def _draw_dynamic_wrapped_text(self, draw, text, box, font_path, fill):
        left, top, right, bottom = box
        max_width = max(10, right - left)
        max_height = max(10, bottom - top)
        
        # Adjust for vertical bubbles: if bubble is very tall, expand width to allow natural Korean horizontal reading
        if max_height > max_width * 1.2:
            expansion = (max_height - max_width) * 0.4
            max_width += expansion
            left -= expansion / 2
            right += expansion / 2
        
        # Start with a proportional font size and shrink until it fits
        font_size = int(max_height * 0.5)  # Start smaller than full height for bubbles
        if font_size < 10:
            font_size = 10
            
        optimal_lines = []
        optimal_font = None
        
        while font_size >= 8:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                try:
                    font = ImageFont.truetype("malgun.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                    
            lines = self._get_wrapped_lines(text, font, max_width, draw)
            
            bbox = draw.textbbox((0, 0), "가", font=font)
            line_height = bbox[3] - bbox[1]
            if line_height <= 0:
                line_height = font_size
            line_spacing = int(line_height * 1.3) # 30% spacing for Korean text
                
            total_height = len(lines) * line_spacing
            
            if total_height <= max_height:
                optimal_lines = lines
                optimal_font = font
                break
                
            font_size -= max(1, int(font_size * 0.1)) # Shrink by 10%
            
        if not optimal_font:
            # Fallback if even size 8 doesn't fit
            optimal_font = font
            optimal_lines = lines
            
        # Refetch line height for the optimal font
        bbox = draw.textbbox((0, 0), "가", font=optimal_font)
        line_height = bbox[3] - bbox[1]
        if line_height <= 0:
            line_height = font_size
        line_spacing = int(line_height * 1.3)
            
        total_height = len(optimal_lines) * line_spacing
        current_y = top + (max_height - total_height) / 2
        
        for line in optimal_lines:
            lw = draw.textlength(line, font=optimal_font)
            current_x = left + (max_width - lw) / 2
            draw.text((current_x, current_y), line, font=optimal_font, fill=fill)
            current_y += line_spacing

    def _get_wrapped_lines(self, text, font, max_width, draw):
        lines = []
        # Respect explicit newlines by splitting first
        for paragraph in text.split('\n'):
            words = paragraph.split()
            if not words:
                lines.append("")
                continue
                
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                w = draw.textlength(test_line, font=font)
                
                if w <= max_width:
                    current_line = test_line
                else:
                    word_w = draw.textlength(word, font=font)
                    if word_w > max_width:
                        if current_line:
                            lines.append(current_line)
                            current_line = ""
                        for char in word:
                            test_line_char = current_line + char
                            char_w = draw.textlength(test_line_char, font=font)
                            if char_w <= max_width:
                                current_line = test_line_char
                            else:
                                if current_line:
                                    lines.append(current_line)
                                current_line = char
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
            if current_line:
                lines.append(current_line)
        return lines
