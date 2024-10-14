import streamlit as st
from transformers import pipeline  # Example using Hugging Face's transformers
import matplotlib.pyplot as plt
from wordcloud import WordCloud
'''
# Call set_page_config only once at the beginningst.set_page_config(page_title="تطبيق تحويل الصوت إلى نص", layout="wide")

# Custom CSS to apply right-to-left layout
st.markdown("""
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)
'''
# Define the show method for the topic modeling view
def show():
    st.title("نمذجة الموضوعات")
    
    # User input: text or document upload
    user_input = st.text_area("أدخل النص أو قم بتحميل مستند لتحليل الموضوعات:")
    
    # Button to trigger topic modeling
    if st.button("تحليل الموضوعات"):
        if user_input:
            # Load a pre-trained model (this is a placeholder, use an actual topic modeling pipeline)
            model = pipeline('text-classification', model='distilbert-base-uncased')
            topics = model(user_input)
            
            # Extract the top topics (for simplicity, assume the model output works similarly)
            st.subheader("الموضوعات المحددة")
            for topic in topics:
                st.write(f"الموضوع: {topic['label']}, النتيجة: {topic['score']:.2f}")
            
            # Visualization: Create a word cloud (replace with real topic modeling output)
            st.subheader("تصور الموضوعات")
            wordcloud = WordCloud(font_path='path_to_arabic_font.ttf').generate(user_input)  # Example word cloud from input text
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt.gcf())
        else:
            st.warning("يرجى إدخال النص أو تحميل مستند.")

