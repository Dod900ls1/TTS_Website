from speech_synthesis import initialize_device, load_model, process_text_to_speech
import nemo.collections.tts as nemo_tts
import IPython.display as ipd
from text_processing import pdf_to_text

def main():
    device = initialize_device()
    tacotron2 = load_model(nemo_tts.models.Tacotron2Model, "tts_en_tacotron2", device)
    hifigan = load_model(nemo_tts.models.HifiGanModel, "tts_en_hifigan", device)
    
    large_text = pdf_to_text("12.pdf")
    
    full_audio = process_text_to_speech(tacotron2, hifigan, device, large_text)
    output_path = "In_Search_Of_The_Miraculous.wav"
    full_audio.export(output_path, format="wav")
    
    return ipd.Audio(output_path)

if __name__ == "__main__":
    main()
