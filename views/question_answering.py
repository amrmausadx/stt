import streamlit as st
from transformers import pipeline

# Define the show method for the question answering view
def show():
    st.title("Question Answering Using Hugging Face Models")
    
    # User input: paragraph or document
    user_text = st.text_area("Input a paragraph or document:")
    
    # User input: question
    user_question = st.text_input("Ask a question based on the text:")
    
    # Button to trigger question answering
    if st.button("Get Answer"):
        if user_text and user_question:
            # Load a pre-trained question-answering model
            qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
            # Use the model to get the answer
            answer = qa_pipeline(question=user_question, context=user_text)
            
            # Display the answer
            st.subheader("Answer:")
            st.write(answer['answer'])
        else:
            st.warning("Please input both the text and a question.")
