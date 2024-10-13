import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf
import torchaudio

MODEL_NAME = "facebook/wav2vec2-base-960h"
TARGET_SAMPLE_RATE = 16000

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

def resample_audio(audio, original_sample_rate):
    if original_sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=TARGET_SAMPLE_RATE)
        audio = resampler(audio)
    return audio

def transcribe_audio(audio_file):
    try:
        audio_input, sample_rate = sf.read(audio_file)
        audio_tensor = torch.tensor(audio_input, dtype=torch.float32)
        audio_tensor = resample_audio(audio_tensor, sample_rate)
        inputs = processor(audio_tensor, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription
    except Exception as e:
        st.error(f"خطأ في معالجة ملف الصوت: {str(e)}")
        return None

def show():
    st.title("تحويل الصوت إلى نص")
    uploaded_file = st.file_uploader("ارفع ملف صوتي", type=["wav", "mp3", "flac"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        if 'transcription' not in st.session_state or st.session_state.uploaded_file != uploaded_file:
            with st.spinner("جاري تحويل الصوت..."):
                transcription = transcribe_audio(uploaded_file)
                if transcription:
                    st.session_state.transcription = transcription
                    st.session_state.uploaded_file = uploaded_file
        transcribed_text = st.text_area("النص المحول", value=st.session_state.get("transcription", ""), height=200)
