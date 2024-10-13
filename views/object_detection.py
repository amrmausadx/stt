import streamlit as st
from transformers import pipeline
from PIL import Image

# Load models
object_detection_model = pipeline("object-detection", model="facebook/detr-resnet-50")
captioning_model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

def detect_objects(image):
    results = object_detection_model(image)
    return results

def generate_caption(image):
    results = captioning_model(image)
    return results[0]['generated_text']  # Adjust according to the model's output format

def show():
    # Set the page config to make it right-to-left
    st.set_page_config(page_title="تطبيق الكشف عن الكائنات والتعليق على الصور", layout="wide")
    
    # Add custom CSS to set right-to-left direction
    st.markdown(
        """
        <style>
        body {
            direction: rtl;
            text-align: right;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.title("تطبيق الكشف عن الكائنات والتعليق على الصور")
    
    uploaded_file = st.file_uploader("قم بتحميل صورة...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='الصورة المحملة', use_column_width=True)
        
        col1, col2 = st.columns(2)  # Create two columns for buttons
        
        with col1:
            if st.button("كشف الكائنات"):
                objects = detect_objects(image)
                st.write("الكائنات المكتشفة:")
                for obj in objects:
                    st.write(f"{obj['label']} بنسبة ثقة {obj['score']:.2f}")

        with col2:
            if st.button("إنشاء وصف"):
                caption = generate_caption(image)
                st.write("الوصف الناتج:")
                st.write(caption)

if __name__ == "__main__":
    show()
