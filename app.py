import streamlit as st
import soundfile as sf
import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set up Wav2Vec2 model and processor
MODEL_NAME = "facebook/wav2vec2-base-960h"  # Use a smaller model
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

st.title("Speech-to-Text App")
st.write("Upload an audio file, and the application will transcribe it using a pre-trained Wav2Vec2 model.")

# Limit the duration of the audio to 60 seconds
max_duration = 60  # Max duration in seconds

def transcribe_audio(file_path):
    try:
        audio_input, sample_rate = sf.read(file_path)
        if len(audio_input) / sample_rate > max_duration:
            st.error(f"Audio file exceeds maximum duration of {max_duration} seconds. Please upload a shorter file.")
            return None

        input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
        with st.spinner("Transcribing..."):
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            return transcription
    except Exception as e:
        st.error(f"Error in processing audio file: {e}")
        return None

# Upload an audio file
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    file_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    
    # Save uploaded file to disk
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(file_path)  # Play the uploaded audio file

    # Transcribe button
    if st.button("Transcribe"):
        transcription = transcribe_audio(file_path)
        if transcription:
            st.write("### Transcription")
            st.text_area("Transcription", transcription)

        # Clear session cache after transcription
        st.caching.clear_cache()

    # Clean up the temp file after use
    if os.path.exists(file_path):
        os.remove(file_path)

# Provide option to clear session manually if needed
if st.button("Clear Session"):
    st.caching.clear_cache()
    st.success("Session cleared!")
