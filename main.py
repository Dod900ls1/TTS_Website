from speech_synthesis import initialize_device, load_model, split_text, generate_spectrogram, convert_to_audio, concatenate_audio, process_text_to_speech
import nltk
import nemo.collections.tts as nemo_tts
import IPython.display as ipd
from text_processing import pdf_to_text

nltk.download('punkt')

def main():
    device = initialize_device()
    tacotron2 = load_model(nemo_tts.models.Tacotron2Model, "tts_en_tacotron2", device)
    hifigan = load_model(nemo_tts.models.HifiGanModel, "tts_en_hifigan", device)
    
    large_text = pdf_to_text("sample_test_pdf.pdf")
    
    full_audio = process_text_to_speech(tacotron2, hifigan, device, large_text)
    output_path = "output_large_text.wav"
    full_audio.export(output_path, format="wav")
    
    return ipd.Audio(output_path)

if __name__ == "__main__":
    main()