import streamlit as st
from transformers import pipeline

# تعريف دالة العرض لعرض سؤال وإجابة
def show():
    st.title("الإجابة على الأسئلة باستخدام نماذج هاجينغ فيس")
    
    # إدخال المستخدم: فقرة أو مستند
    user_text = st.text_area("أدخل فقرة أو مستند:")
    
    # إدخال المستخدم: السؤال
    user_question = st.text_input("اسأل سؤالاً بناءً على النص:")
    
    # زر لتفعيل الإجابة على السؤال
    if st.button("احصل على إجابة"):
        if user_text and user_question:
            # تحميل نموذج إجابة على الأسئلة مسبق التدريب
            qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
            # استخدام النموذج للحصول على الإجابة
            answer = qa_pipeline(question=user_question, context=user_text)
            
            # عرض الإجابة
            st.subheader("الإجابة:")
            st.write(answer['answer'])
        else:
            st.warning("يرجى إدخال كل من النص والسؤال.")
