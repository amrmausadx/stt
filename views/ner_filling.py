import streamlit as st
from transformers import pipeline
from googletrans import Translator

def show():
    st.title("التعرف على الكيانات وإكمال الجمل والترجمة")
    
    ner_model_name = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"
    try:
        ner_model = pipeline("ner", model=ner_model_name)
    except Exception as e:
        st.error(f"خطأ في تحميل نموذج التعرف على الكيانات: {e}")
        ner_model = pipeline("ner", model="asafaya/bert-base-arabic")

    sentence_completion_model = pipeline("fill-mask", model="asafaya/bert-base-arabic")
    text = st.text_area("أدخل نصًا", height=100)
    
    # Create three columns for buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("التعرف على الكيانات"):
            if text:
                entities = ner_model(text)
                st.subheader("الكيانات المعترف بها:")
                for entity in entities:
                    st.write(f"**{entity['word']}**: {entity['entity']} (النسبة: {entity['score']:.4f})")

    with col2:
        if st.button("إكمال الجملة"):
            if "[MASK]" in text:
                completions = sentence_completion_model(text)
                st.subheader("إكمال الجمل:")
                for completion in completions:
                    st.write(f"الخيار: **{completion['sequence']}** (النسبة: {completion['score']:.4f})")

    with col3:
        translator = Translator()
        language_options = {"الإنجليزية": "en", "الفرنسية": "fr", "الصينية": "zh-cn", "العبرية": "he"}
        selected_language = st.selectbox("اختر لغة للترجمة", list(language_options.keys()))
        
        if st.button("ترجمة"):
            if text:
                target_language = language_options[selected_language]
                translation = translator.translate(text, dest=target_language)
                st.write(f"الترجمة: **{translation.text}**")

