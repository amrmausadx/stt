import streamlit as st
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np
import tempfile
import os

# Initialize the model and processor
@st.cache_resource
def load_model():
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    return model, processor

# Function to convert speech to text
def transcribe_audio(audio_file, model, processor):
    audio, rate = librosa.load(audio_file, sr=16000)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

# Streamlit App Layout
st.title("Speech to Text App")
st.write("Upload an audio file or record your voice to get the transcription.")

# Load Model and Processor
model, processor = load_model()

# File uploader
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Real-time audio recording using streamlit_webrtc (optional feature if installed)
try:
    from streamlit_webrtc import webrtc_streamer
    import av

    def audio_frame_callback(frame: av.AudioFrame):
        sound_array = frame.to_ndarray()
        # Save audio frames to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            librosa.output.write_wav(tmp_file.name, sound_array, sr=16000)
            return tmp_file.name

    webrtc_ctx = webrtc_streamer(key="speech-recorder", audio_frame_callback=audio_frame_callback, media_stream_constraints={"audio": True})

    if webrtc_ctx.audio_receiver:
        st.write("Recording... Click the stop button when done.")

except ImportError:
    st.warning("Install streamlit-webrtc to enable the real-time recording feature.")

# Process uploaded or recorded audio
if audio_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        file_path = tmp_file.name

    st.audio(audio_file, format='audio/wav')

    # Transcribe the uploaded audio
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(file_path, model, processor)
            st.text_area("Transcription", transcription)

# Record transcription for recorded audio
if webrtc_ctx and webrtc_ctx.audio_receiver:
    if webrtc_ctx.state.playing == False and audio_frame_callback:
        if st.button("Transcribe Recorded Audio"):
            with st.spinner("Transcribing..."):
                transcription = transcribe_audio(webrtc_ctx.audio_receiver.audio_buffer, model, processor)
                st.text_area("Transcription", transcription)

# Clear temporary files
if os.path.exists(file_path):
    os.remove(file_path)
