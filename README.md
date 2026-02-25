# BASINC TRANSLATER (만화 식자 번역 AI)

BASINC TRANSLATER는 최신 AI 기술들을 결합하여 일본어 만화 이미지를 한국어로 자연스럽게 번역해 주는 혁신적인 자동 식자(Typesetting) 프로그램입니다.
오프라인 로컬 객체 탐지 모델(Comic Text Detector)의 정밀함과 구글 Gemini Pro 비전 언어 모델의 문맥 추론 능력을 결합하여, 단순 번역을 넘어 캐릭터의 톤앤매너와 장르까지 반영한 고품질 번역을 제공합니다.

## ✨ 주요 기능 (Key Features)

### 1. 듀얼-박스(Dual-Box) 정밀 지우개 기법
- 기존의 단순한 OCR 번역기나 AI들이 네모난 말풍선 영역 전체를 통째로 지워버려 배경 아트워크가 손상되던 문제를 완벽히 해결했습니다.
- **`Comic Text Detector`** 로컬 모델이 텍스트의 "정확한 글씨 세로선(Stroke)"만을 찾아내고, 라마(LaMa) 기반의 AI 인페인팅 모듈이 주변 그림을 손상시키지 않은 채 오로지 텍스트 픽셀만 감쪽같이 지워냅니다. 말풍선 밖의 배경 일러스트도 원본 그대로 보존됩니다.

### 2. 구글 Gemini 기반의 장르/문맥 맞춤 번역
- 단순 기계 번역이 아닙니다. Gemini AI가 이미지 전체의 화풍, 장르, 캐릭터의 표정과 상황을 우선적으로 분석합니다.
- 분석을 바탕으로 캐릭터 간의 존댓말/반말, 구어체, 성격을 모두 반영한 **상황 맞춤형 한국어 번역**을 생성해 자연스러운 대화문으로 만듭니다.

### 3. 효과음(의성어/의태어) 보존 시스템 (SFX Pruning)
- 만화책 특유의 생동감을 살리는 효과음(예: 랄랄라, 맴맴, 쾅, 휙 등)은 번역하거나 지우지 않는 것이 오히려 몰입감에 좋습니다.
- 번역 AI가 효과음으로 인식한 텍스트는 인페인팅(지우기)과 오버레이(새로 쓰기) 단계에서 완전히 배제되어, 작가가 그린 원래의 폰트와 아트 그대로 유지됩니다.

### 4. 자연스러운 한국어 가로 쓰기 동적 렌더링
- 일본어 특유의 좁고 긴 세로 쓰기 말풍선에 한국어를 가로로 욱여넣다 보면 단어가 심하게 끊어지는 문제가 생깁니다.
- 프로그램이 말풍선의 형태(가로세로 비율)를 자체 분석하여, 극단적으로 좁은 세로 말풍선일 경우 **한국어 텍스트가 자연스럽게 들어갈 수 있도록 렌더링 폭을 동적으로 최대 40%까지 확장**합니다. 줄바꿈을 최소화하여 가독성을 극대화했습니다.

## 🚀 설치 및 실행 방법 (Installation & Usage)

### 사전 준비 (Prerequisites)
- Python 3.9 이상
- Google Gemini API Key (`dotenv`를 통해 로드)
- C++ Build Tools (IOPaint 및 일부 라이브러리 설치용)

### 1) 저장소 클론 및 패키지 설치
```bash
git clone https://github.com/jskim062/BASINC_TRANSLATER.git
cd BASINC_TRANSLATER
pip install -r requirements.txt
```

### 2) 환경 변수 설정
프로젝트 최상단 폴더에 `.env` 파일을 만들고 아래와 같이 작성합니다:
```ini
GEMINI_API_KEY=여러분의_제미나이_API_키를_여기에_입력하세요
```

### 3) 폰트 파일 준비 (선택)
본 레포지토리에 기본적으로 `NanumSquareRound` 폰트가 내장되어 있습니다. 만약 다른 폰트를 사용하고 싶다면 프로젝트 폴더에 `.ttf` 혹은 `.otf` 폰트 파일을 넣고 코드 내 폰트 경로를 수정하세요.

### 4) Comic Text Detector 모델 다운로드
원활한 객체 탐지를 위해 사전 학습된 ONNX 모델 파일이 필요합니다.
- `comictextdetector.pt.onnx` 형태의 모델 파일을 구하여 프로젝트 폴더 직속에 위치시킵니다. (용량 문제로 GitHub에는 올라가 있지 않습니다.)

### 5) 프로그램 실행
```bash
streamlit run app.py
```
브라우저 창이 열리면, 번역하고 싶은 만화 페이지 이미지를 업로드하고 **[Translate Component]** 버튼을 누르시면 수십 초 내에 번역된 이미지가 나타납니다.

## 🛠 기술 스택 (Tech Stack)
- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Text Detection:** [Comic Text Detector (YOLO-style Object Detection) via ONNX Runtime](https://github.com/dmMaze/comic-text-detector)
- **Translation & OCR:** [Google Gemini Pro Vision API](https://deepmind.google/technologies/gemini/)
- **Inpainting (Text Removal):** [IOPaint (LaMa Model)](https://github.com/Sanster/IOPaint)
- **Image Processing:** OpenCV (`cv2`), Pillow (`PIL`), NumPy

## 📝 라이선스 (License)
본 프로젝트에 대한 사용 권한은 레포지토리 소유자에게 있습니다. 모델 추론용으로 사용된 서드파티 라이브러리들의 라이선스 규정을 준수합니다.
