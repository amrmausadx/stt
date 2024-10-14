import streamlit as st
from transformers import pipeline

# تعريف دالة العرض لواجهة أداة ترجمة اللغة وتصحيح القواعد
def show():
    st.title("أداة ترجمة اللغة وتصحيح القواعد")
    
    # إدخال المستخدم: نص للترجمة أو التصحيح
    user_text = st.text_area("أدخل النص لترجمته أو تصحيحه:")
    
    # إدخال المستخدم: اختيار اللغة المستهدفة للترجمة
    target_language = st.selectbox("اختر اللغة المستهدفة للترجمة:", 
                                    ["الإنجليزية", "العربية", "الفرنسية", "الإسبانية", "الألمانية"])
    
    # أزرار للترجمة وتصحيح القواعد
    if st.button("ترجمة"):
        if user_text:
            # تحميل نموذج ترجمة مدرب مسبقًا
            translation_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")  # مثال للترجمة من الإنجليزية إلى العربية
            translation = translation_model(user_text)
            
            # عرض النص المترجم
            st.subheader("النص المترجم:")
            st.write(translation[0]['translation_text'])
        else:
            st.warning("يرجى إدخال نص للترجمة.")
    
    if st.button("فحص القواعد"):
        if user_text:
            # تحميل نموذج تصحيح القواعد المدرب مسبقًا
            grammar_correction_model = pipeline("text2text-generation", model="t5-base")  # مثال لتصحيح القواعد
            corrected_text = grammar_correction_model(f"تصحيح القواعد: {user_text}")
            
            # عرض النص المصحح
            st.subheader("النص المصحح:")
            st.write(corrected_text[0]['generated_text'])
        else:
            st.warning("يرجى إدخال نص لتصحيح القواعد.")
