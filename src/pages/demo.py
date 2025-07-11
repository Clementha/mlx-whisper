import streamlit as st
import torch
import whisper
from utils import get_device, load_artifact_path, init_wandb

MODEL_VERSION = "v0"

@st.cache_resource(show_spinner=False)
def load_models():
    init_wandb()
    device = get_device()
    model_file_path = load_artifact_path("fine_tuned_welsh_model", MODEL_VERSION, "pth")

    # Load default (pretrained) Whisper model
    default_model = whisper.load_model("tiny", device=device)
    default_model.to(device)
    default_model.eval()

    # Load fine-tuned Whisper model
    fine_tuned_model = whisper.load_model("tiny", device=device)
    fine_tuned_model.load_state_dict(torch.load(model_file_path, map_location=device))
    fine_tuned_model.to(device)
    fine_tuned_model.eval()

    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

    return default_model, fine_tuned_model, tokenizer, device

def transcribe_audio(model, device, audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)

    options = whisper.DecodingOptions(language=None, task="transcribe", without_timestamps=True)
    result = whisper.decode(model, mel, options)

    return result.text, result.language

def main():
    st.title("Welsh audio transcription")
    st.subheader("Default tiny whisper vs fine-tuned")

    # Fixed test audio file path
    test_audio_path = "src/test.wav"

    # Play the audio file
    st.audio(test_audio_path)

    with st.spinner("Loading models, please wait..."):
        default_model, fine_tuned_model, tokenizer, device = load_models()

    st.write("Transcribing...")

    default_text, default_lang = transcribe_audio(default_model, device, test_audio_path)
    tuned_text, tuned_lang = transcribe_audio(fine_tuned_model, device, test_audio_path)

    st.success("Transcription complete!")

    st.subheader("Default Whisper Transcription")
    st.write(f"Detected language: `{default_lang}`")
    st.text_area("Default", default_text, height=150)

    st.subheader("Fine-Tuned Whisper Transcription")
    st.write(f"Detected language: `{tuned_lang}`")
    st.text_area("Fine-Tuned", tuned_text, height=150)

    st.subheader("Actual")
    st.text("Mae'n boeth iawn yn yr ostafell hon.")

if __name__ == "__main__":
    main()
