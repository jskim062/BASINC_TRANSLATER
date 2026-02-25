import streamlit as st
import os
from translator import MangaTranslator
from image_processor import ImageProcessor
from detector import ComicTextDetector
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Gemini Manga Translator", layout="wide")

st.title("📖 Gemini Manga Translator (JP ➡ KR)")
st.write("Translate Japanese manga pages to Korean using Gemini 2.5 Pro.")

@st.cache_resource
def get_detector():
    return ComicTextDetector()

# Get API Key from environment first
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    # Sidebar for API Key if not in .env
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    
if not api_key:
    st.warning("Please provide a Gemini API Key in the sidebar or .env file.")
else:
    try:
        translator = MangaTranslator(api_key=api_key)
        processor = ImageProcessor()
        
        uploaded_file = st.file_uploader("Upload a Manga Page (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info("Detecting and translating text... This may take a moment.")
            
            cols = st.columns(2)
            
            # Show original
            original_img = Image.open(temp_path)
            cols[0].image(original_img, caption="Original Page", use_container_width=True)
            
            use_local_detector = st.checkbox("Use High Precision Text Detector (Comic Text Detector)", value=True)
            
            # Translate and Process
            with st.spinner("Gemini is processing..."):
                detected_items = None
                raw_boxes = None
                if use_local_detector:
                    detector = get_detector()
                    detected_items, raw_boxes = detector.detect(temp_path)
                    
                image_bytes = uploaded_file.getbuffer().tobytes()
                translations = translator.translate_page(
                    image_bytes=image_bytes,
                    mime_type=uploaded_file.type,
                    detected_items=detected_items
                )

                
                if translations:
                    # Handle both old list format and new dict format
                    translation_list = translations.get("translations", []) if isinstance(translations, dict) else translations
                    analysis = translations.get("analysis", {}) if isinstance(translations, dict) else {}
                    
                    if analysis:
                        st.info("💡 **Analysis:**")
                        st.write(f"- **Genre:** {analysis.get('genre', 'N/A')}")
                        st.write(f"- **Characters/Tone:** {analysis.get('characters', 'N/A')} | {analysis.get('tone', 'N/A')}")
                        
                    st.success(f"Detected {len(translation_list)} text items.")
                    processed_img = processor.process_manga_page(temp_path, translation_list, raw_boxes)
                    cols[1].image(processed_img, caption="Translated Page", use_container_width=True)
                    
                    # Option to download
                    st.download_button(
                        label="Download Translated Image",
                        data=uploaded_file.getbuffer(), # Placeholder - needs actual save
                        file_name=f"translated_{uploaded_file.name}",
                        mime="image/png"
                    )
                else:
                    st.error("No text detected or error in translation.")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        st.error(f"Error: {e}")
