import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

# Load pre-trained model and processor
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# Load the audio file (you can adapt this for your Streamlit file upload)
audio, rate = librosa.load("path_to_audio.wav", sr=16000)

# Pre-process audio
input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

# Perform inference
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)

# Decode the result 
transcription = processor.decode(predicted_ids[0])
print(transcription)
