import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf
import torchaudio

# Load pre-trained model and processor
MODEL_NAME = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

# Set the expected sample rate for the model
TARGET_SAMPLE_RATE = 16000

st.title("Speech to Text Application")

def resample_audio(audio, original_sample_rate):
    if original_sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=TARGET_SAMPLE_RATE)
        audio = resampler(audio)
    return audio

# Function to convert speech to text
def transcribe_audio(audio_file):
    try:
        # Load the audio file
        audio_input, sample_rate = sf.read(audio_file)

        # Convert numpy audio data to tensor and resample if needed
        audio_tensor = torch.tensor(audio_input, dtype=torch.float32)  # Ensure it's float32
        audio_tensor = resample_audio(audio_tensor, sample_rate)

        # Process audio and get model prediction
        inputs = processor(audio_tensor, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription
    except Exception as e:
        st.error(f"Error in processing audio file: {str(e)}")
        return None

# Sidebar with navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a view", ["Upload Audio", "View Transcription", "About"])

# View 1: Upload and Transcribe Audio
if option == "Upload Audio":
    st.header("Upload and Transcribe Audio")
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Only transcribe if the uploaded file has changed
        if 'transcription' not in st.session_state or st.session_state.uploaded_file != uploaded_file:
            with st.spinner("Transcribing audio..."):
                transcription = transcribe_audio(uploaded_file)
                if transcription:
                    st.session_state.transcription = transcription  # Store the transcription in session state
                    st.session_state.uploaded_file = uploaded_file  # Store the uploaded file

# View 2: Display Transcription
elif option == "View Transcription":
    st.header("View Transcription")
    
    if 'transcription' in st.session_state:
        # Display transcription in a wide text area
        transcribed_text = st.text_area("Transcription", value=st.session_state.get("transcription", ""), height=200)
    else:
        st.write("No transcription available. Please upload and transcribe an audio file first.")

# View 3: About Section
elif option == "About":
    st.header("About This Application")
    st.write("""
        This Speech-to-Text application allows users to upload audio files or record their voice, 
        and converts the speech into text using Hugging Face's Wav2Vec2 model. 
        The app is built using Streamlit and deployed with continuous integration via GitHub Actions.
    """)
    st.write("""
        ### Features:
        - Upload audio files in WAV, MP3, or FLAC format.
        - View transcriptions of uploaded audio.
        - Integrated with pre-trained models from Hugging Face for state-of-the-art speech recognition.
    """)

