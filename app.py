import streamlit as st
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import os
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, ClientSettings

# Load pre-trained model and processor from Hugging Face
MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

st.title("Speech to Text App")

# Function to transcribe audio files
def transcribe_audio(file_path):
    # Load audio file
    audio_input, _ = sf.read(file_path)
    
    # Tokenize the input audio
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values

    # Get logits from model
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
if uploaded_file:
    file_path = os.path.join("tempDir", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Transcribe uploaded audio
    st.audio(file_path)
    if st.button("Transcribe Audio File"):
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(file_path)
        st.text_area("Transcription", transcription)

# WebRTC for real-time recording and transcription
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = b''

    def recv(self, frame):
        self.buffer += frame.to_ndarray().tobytes()
        return frame

webrtc_ctx = webrtc_streamer(
    key="example", 
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
    ),
    audio_processor_factory=AudioProcessor,
)

if webrtc_ctx and webrtc_ctx.state.playing:
    if st.button("Transcribe Recorded Audio"):
        with st.spinner("Transcribing recorded audio..."):
            audio_data = webrtc_ctx.audio_processor.buffer
            if audio_data:
                file_path = "tempDir/recorded_audio.wav"
                with open(file_path, "wb") as f:
                    f.write(audio_data)
                transcription = transcribe_audio(file_path)
                st.text_area("Transcription", transcription)
            else:
                st.warning("No audio recorded yet.")
