import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress all warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Just shup up

# Import the view modules
from views import transcribe, ner_filling, about, object_detection

# Set page configuration at the very beginning
st.set_page_config(page_title="تطبيق تحويل الصوت إلى نص", layout="wide")

st.markdown(
    """
    <style>
    body {
        direction: rtl !important;
        text-align: right !important;
    }
    .stTextInput, .stButton, .stFileUploader, .stTextArea {
        direction: rtl !important;
        text-align: right !important;
    }
    </style>
    """, 
    unsafe_allow_html=True
)
mymenu = ['تحويل الصوت إلى نص',
                                         'التعرف على الكيانات وإكمال الجمل والترجمة',
                                          'التعرف على الاشياء',
                                            'حول']
# Sidebar with options
menu = st.sidebar.selectbox("القائمة", mymenu)

# Call the appropriate view based on the selection
if menu == mymenu[0]:# 'تحويل الصوت إلى نص'
    transcribe.show()

elif menu == mymenu[1]:#'التعرف على الكيانات وإكمال الجمل والترجمة':
    ner_filling.show()

elif menu == mymenu[2]:#'التعرف على الاشياء':
    object_detection.show()

elif menu == mymenu[3]:#'حول':
    about.show()
