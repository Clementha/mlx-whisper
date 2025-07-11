import streamlit as st
import torch
import whisper
import tempfile
import mimetypes
import os
from utils import get_device, load_artifact_path, init_wandb
from translate import Translator

MODEL_VERSION = "v0"

@st.cache_resource(show_spinner=False)
def load_models():
    init_wandb()
    device = get_device()
    model_file_path = load_artifact_path("fine_tuned_welsh_model", MODEL_VERSION, "pth")

    # Load default Whisper model
    default_model = whisper.load_model("tiny", device=device)
    default_model.to(device)
    default_model.eval()

    # Load fine-tuned model
    fine_tuned_model = whisper.load_model("tiny", device=device)
    fine_tuned_model.load_state_dict(torch.load(model_file_path, map_location=device))
    fine_tuned_model.to(device)
    fine_tuned_model.eval()

    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
    return default_model, fine_tuned_model, tokenizer, device

def save_uploaded_file(uploaded_file):
    """Save UploadedFile or audio_input to a temporary file."""
    file_type = uploaded_file.type or "audio/wav"
    suffix = mimetypes.guess_extension(file_type) or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def transcribe_audio(model, device, audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)

    options = whisper.DecodingOptions(task="transcribe", without_timestamps=True)
    result = whisper.decode(model, mel, options)

    return result.text, result.language

def translate_to_welsh(text):
    translator = Translator(to_lang="cy")
    try:
        return translator.translate(text)
    except Exception as e:
        return f"Translation error: {e}"

def main():
    # Expandable English-to-Welsh translation at top
    st.title("Say: llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch")

    audio_value = st.audio_input("Record a voice message")

    if audio_value:
        st.audio(audio_value, format=audio_value.type)
        st.write("Processing your recorded audio...")

        try:
            file_path = save_uploaded_file(audio_value)

            with st.spinner("Loading models, please wait..."):
                default_model, fine_tuned_model, tokenizer, device = load_models()

            st.write("Transcribing...")
            default_text, default_lang = transcribe_audio(default_model, device, file_path)
            tuned_text, tuned_lang = transcribe_audio(fine_tuned_model, device, file_path)

            os.remove(file_path)

            st.success("Transcription complete!")

            st.subheader("Default Whisper Transcription")
            st.write(f"Detected language: `{default_lang}`")
            st.text_area("Default", default_text, height=150)

            st.subheader("Fine-Tuned Whisper Transcription")
            st.write(f"Detected language: `{tuned_lang}`")
            st.text_area("Fine-Tuned", tuned_text, height=150)

            st.subheader("Actual")
            st.text("llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch")

        except Exception as e:
            st.error(f"Transcription failed: {e}")

if __name__ == "__main__":
    main()
