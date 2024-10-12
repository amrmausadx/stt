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

# Upload audio file or record audio
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with st.spinner("Transcribing audio..."):
        transcription = transcribe_audio(uploaded_file)
        if transcription:
            # Display transcription in a wide text area
            transcribed_text = st.text_area("Transcription", value=transcription, height=200)

            # Button to copy to clipboard
            if st.button("Copy to clipboard"):
                # Use Streamlit's session state to store the transcription
                st.session_state.transcription = transcription
                
                # JavaScript code for copying to clipboard
                st.markdown(f"""
                <script>
                navigator.clipboard.writeText("{st.session_state.transcription}").then(function() {{
                    alert("Transcription copied!");
                }});
                </script>
                """, unsafe_allow_html=True)
