import streamlit as st
from transformers import pipeline

# Define the view for language translation and grammar correction
def show():
    st.title("أداة ترجمة اللغة وتصحيح القواعد")
    
    # User input: text to translate or correct
    user_text = st.text_area("أدخل النص لترجمته أو تصحيحه:")
    
    # User input: select translation target language
    target_language = st.selectbox("اختر اللغة المستهدفة للترجمة:", 
                                    ["الإنجليزية", "العربية", "الفرنسية", "الإسبانية", "الألمانية"])
    
    # Buttons for translation and grammar correction
    if st.button("ترجمة"):
        if user_text:
            # Load a pre-trained translation model (example: English to Arabic)
            translation_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")
            translation = translation_model(user_text)
            
            # Display the translated text
            st.subheader("النص المترجم:")
            st.write(translation[0]['translation_text'])
        else:
            st.warning("يرجى إدخال نص للترجمة.")
    
    if st.button("فحص القواعد"):
        if user_text:
            # Load a pre-trained grammar correction model
            grammar_correction_model = pipeline("text2text-generation", model="t5-base")
            
            # Format the input for grammar correction task
            corrected_text = grammar_correction_model(f"fix grammar: {user_text}")
            
            # Display the corrected text
            st.subheader("النص المصحح:")
            st.write(corrected_text[0]['generated_text'])
        else:
            st.warning("يرجى إدخال نص لتصحيح القواعد.")
