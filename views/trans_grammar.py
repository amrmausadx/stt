import streamlit as st
from transformers import pipeline

# Define the view for language translation and grammar correction
def show():
    st.title("أداة ترجمة اللغة من الانجليزية وتصحيح القواعد")
    
    # User input: text to translate or correct
    user_text = st.text_area("أدخل النص لترجمته أو تصحيحه:")
    
    # User input: select translation target language
    target_language = st.selectbox("اختر اللغة المستهدفة للترجمة:", 
                                    ["English to Arabic", "English to French", "English to Spanish", "English to German"])
    
    # Buttons for translation and grammar correction
    if st.button("ترجمة"):
        if user_text:
            # Choose the translation model based on the target language
            if target_language == "English to Arabic":
                translation_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")
            elif target_language == "English to French":
                translation_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
            elif target_language == "English to Spanish":
                translation_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
            elif target_language == "English to German":
                translation_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
                
            # Perform the translation
            translation = translation_model(user_text)
            
            # Display the translated text
            st.subheader("النص المترجم:")
            st.write(translation[0]['translation_text'])
        else:
            st.warning("يرجى إدخال نص للترجمة.")
    
    if st.button("فحص القواعد"):
        if user_text:
            # Load a grammar correction model
            grammar_correction_model = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
            
            # Pass the user text directly
            corrected_text = grammar_correction_model(user_text)
            
            # Display the corrected text
            st.subheader("النص المصحح:")
            st.write(corrected_text[0]['generated_text'])
        else:
            st.warning("يرجى إدخال نص لتصحيح القواعد.")
