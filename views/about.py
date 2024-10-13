import streamlit as st

def show():
    st.title("حول التطبيق")
    st.write("""
        هذا التطبيق يقوم بتحويل الصوت إلى نص باستخدام نماذج مدربة مسبقًا مثل Wav2Vec2 من Hugging Face.
        بالإضافة إلى ذلك، يوفر التعرف على الكيانات المسماة (NER)، إكمال الجمل، وترجمة النصوص.
    """)
