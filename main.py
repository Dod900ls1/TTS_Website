import torch
import nemo.collections.tts as nemo_tts
import IPython.display as ipd
import soundfile as sf

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained FastPitch model
model = nemo_tts.models.FastPitchModel.from_pretrained("tts_en_fastpitch")

# Ensure the model is on the correct device
model.to(device)

# Prepare the text input
text ="Amidst the swirling chaos of life's uncertainties, she pondered: 'Will tomorrow bring hope or despair?' The crisp autumn air carried a hint of nostalgia, mingling with the whispers of time."


# Tokenize the input text
tokens = model.parse(text)

# Generate mel spectrogram from the tokens
with torch.no_grad():
    spectrogram = model.generate_spectrogram(tokens=tokens.to(device))

# Load the HiFi-GAN vocoder using the correct model name from the list
vocoder = nemo_tts.models.HifiGanModel.from_pretrained("tts_en_hifigan")

# Ensure the vocoder is on the correct device
vocoder.to(device)

# Convert the mel spectrogram to audio waveform
with torch.no_grad():
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

# Save the audio to a file
output_path = "output.wav"
audio_numpy = audio.squeeze().cpu().numpy()
sf.write(output_path, audio_numpy, 22050)

# Play the generated audio
ipd.Audio(output_path)

