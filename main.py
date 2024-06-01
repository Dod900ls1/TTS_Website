import torch
import nemo.collections.tts as nemo_tts
import IPython.display as ipd
import soundfile as sf
import nltk
nltk.download('punkt')

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained Tacotron2 model
tacotron2 = nemo_tts.models.Tacotron2Model.from_pretrained(model_name="tts_en_tacotron2")

# Ensure the Tacotron2 model is on the correct device
tacotron2.to(device)

# Load the pre-trained HiFi-GAN vocoder
hifigan = nemo_tts.models.HifiGanModel.from_pretrained(model_name="tts_en_hifigan")

# Ensure the HiFi-GAN model is on the correct device
hifigan.to(device)

# Prepare the large text input
large_text = """
Amidst the swirling chaos of life's uncertainties, she pondered: 'Will tomorrow bring hope or despair?' 
The crisp autumn air carried a hint of nostalgia, mingling with the whispers of time. 
Every leaf that fell seemed to carry a story, a fragment of the past, echoing the transient nature of existence. 
She walked along the path, each step a reflection of the journey, each breath a testament to resilience.
"""

# Tokenize the input text
tokens = tacotron2.parse(large_text)

# Generate mel-spectrogram from the tokens
with torch.no_grad():
    mel_spectrogram = tacotron2.generate_spectrogram(tokens=tokens.to(device))

# Convert the mel-spectrogram to audio waveform
with torch.no_grad():
    audio = hifigan.convert_spectrogram_to_audio(spec=mel_spectrogram)

# Save the audio to a file
output_path = "output_large_text.wav"
audio_numpy = audio.squeeze().cpu().numpy()
sf.write(output_path, audio_numpy, 22050)

# Play the generated audio
ipd.Audio(output_path)
