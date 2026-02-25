import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

class MangaTranslator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in .env file or pass it to constructor.")
        
        genai.configure(api_key=self.api_key)
        # Using gemini-2.0-flash-thinking-exp or similar for advanced reasoning
        # Note: gemini-2.5-pro might be available via specific model names
        self.model_name = "gemini-2.5-pro" # Placeholder for experimental/thinking model
        self.model = genai.GenerativeModel(self.model_name)

    def translate_page(self, image_bytes, mime_type="image/jpeg", detected_items=None):
        """
        Detects text in the manga page and translates it.
        If detected_items is provided, uses those boxes as a guide.
        Returns a list of dictionaries with detected text, translated text, and bounding boxes.
        """
        image_part = {
            "mime_type": mime_type,
            "data": image_bytes
        }
        
        if detected_items:
            # Assign an ID to each detected item to guarantee mapping
            for idx, item in enumerate(detected_items):
                item["id"] = idx + 1
                
            boxes_context = json.dumps([{"id": item["id"], "box_2d": item["box_2d"]} for item in detected_items])
            prompt = f"""
            [시스템 지침]
            너는 만화 전문 번역가이자 콘텐츠 분석가야. 번역을 시작하기 전에 다음 단계를 거쳐줘.

            1단계: 이미지 분석 및 추론
            - 이 만화의 전체적인 장르(코믹, 로맨스, 액션, 스릴러 등)를 그림체와 연출로 파악해.
            - 등장인물들의 외양과 표정을 보고 각각의 성격과 현재 감정 상태를 추론해.
            - 인물들 간의 관계(친구, 적, 연인, 상하관계 등)를 파악해 말투(반말/존댓말)를 결정해.

            2단계: 맞춤형 번역
            - 1단계에서 추론한 결과를 바탕으로, 캐릭터의 개성이 살아있는 자연스러운 한국어로 번역해.
            - 상황에 맞는 적절한 문체(구어체, 신조어, 고어 등)를 선택해.

            제가 이미 텍스트 말풍선 영역(bounding boxes)을 찾아두었습니다.
            각 영역의 고유 ID와 좌표 정보 [ymin, xmin, ymax, xmax] (0-1000 정규화 형태)는 다음과 같습니다:
            {boxes_context}
            
            위의 제공된 각 말풍선 ID에 대해서 텍스트를 읽고 다음 정보를 제공해 주세요:
            1. Original Japanese text
            2. Translated Korean text
            3. 제공받았던 말풍선의 고유 ID (이 ID를 기준으로 매칭할 것입니다).
            4. Estimated font size (0-1000 scale).
            5. Visual style: "standard" or "emphasis".
            
            [제약 조건 (Constraint)]
            - 번역문은 원문 영어/일본어의 글자 수보다 1.2배 이상 길어지지 않게 해줘.
            - 랄랄라, 맴맴, 쾅, 휙 같은 배경 효과음(의성어/의태어)이나 말풍선 밖의 단순 꾸밈 글자로 판단되는 경우, 절대 번역하지 말고 `translated`와 `formatted_text` 값을 빈 문자열 `""`로 반환해.
            
            [출력 형식]
            결과는 반드시 아래의 JSON 형식을 지켜줘(마크다운 백틱 없이 순수 JSON만 반환):
            {{
              "analysis": {{
                "genre": "...",
                "characters": "...",
                "tone": "..."
              }},
              "translations": [
                {{
                  "id": 1,
                  "original": "...",
                  "translated": "...",
                  "formatted_text": "줄바꿈이\n포함된 버전",
                  "font_size": 15,
                  "style": "standard"
                }}
              ]
            }}
            """
        else:
            prompt = """
            Analyze this manga page. Detect every text bubble or caption.
            For each bubble/caption, provide:
            1. Original Japanese text
            2. Translated Korean text
            3. Bounding box coordinates [ymin, xmin, ymax, xmax] in normalized coordinates (0-1000).
            4. Estimated font size (relative to the image height, where 1000 is the total height).
            5. Visual style: "standard" for normal dialogue/narration, or "emphasis" for shouting, sound effects, or stylized calligraphy.
            
            Format the output as a JSON list of objects:
            [
              {
                "original": "...",
                "translated": "...",
                "box_2d": [ymin, xmin, ymax, xmax],
                "font_size": <integer>,
                "style": "standard" | "emphasis"
              },
              ...
            ]
            
            Do not include any Markdown formatting or backticks around the JSON. Only return the raw JSON list.
            """
        
        response = self.model.generate_content([image_part, prompt])
        
        try:
            # Clean up potential markdown formatting in response
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
            
            return_data = json.loads(text)
            
            # Merge original exact box_2d back using ID to prevent Gemini hallucination
            if detected_items and isinstance(return_data, dict) and "translations" in return_data:
                id_to_box = {item["id"]: item["box_2d"] for item in detected_items}
                for t in return_data["translations"]:
                    t_id = t.get("id")
                    if t_id in id_to_box:
                        t["box_2d"] = id_to_box[t_id]
            
            return return_data
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            print(f"Raw response: {response.text}")
            return []
