import nltk
import nemo.collections.tts as nemo_tts
import torch
import soundfile as sf
from pydub import AudioSegment
import os
from typing import List

nltk.download('punkt')

def initialize_device():
    # Initialize the device (e.g., 'cuda' or 'cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_class, model_name, device):
    # Load the specified model
    return model_class.from_pretrained(model_name).to(device)

def split_text(text, max_length=500):
    # Split text into smaller chunks
    return nltk.sent_tokenize(text)

def generate_spectrogram(tacotron2, text, device):
    # Generate a spectrogram from the text
    parsed_text = tacotron2.parse(text)
    with torch.no_grad():
        spectrogram = tacotron2.generate_spectrogram(tokens=parsed_text)
    return spectrogram

def convert_to_audio(hifigan, spectrogram: torch.Tensor, temp_paths: List[str]) -> AudioSegment:
    with torch.no_grad():
        audio = hifigan.convert_spectrogram_to_audio(spec=spectrogram)
    audio_numpy = audio.squeeze().cpu().numpy()
    temp_path = "temp_{}.wav".format(len(temp_paths))  # Unique temp file name
    sf.write(temp_path, audio_numpy, 22050)
    temp_paths.append(temp_path)  # Store temp file path
    return AudioSegment.from_wav(temp_path)

def concatenate_audio(segments: List[AudioSegment], temp_paths: List[str]) -> AudioSegment:
    concatenated_audio = sum(segments, AudioSegment.empty())
    # Delete all temporary files
    for temp_path in temp_paths:
        os.remove(temp_path)
    return concatenated_audio

def process_text_to_speech(tacotron2, hifigan, device, text):
    temp_paths = []
    text_chunks = split_text(text)
    audio_segments = []

    for chunk in text_chunks:
        spectrogram = generate_spectrogram(tacotron2, chunk, device)
        audio_segment = convert_to_audio(hifigan, spectrogram, temp_paths)
        audio_segments.append(audio_segment)

    full_audio = concatenate_audio(audio_segments, temp_paths)
    return full_audio
