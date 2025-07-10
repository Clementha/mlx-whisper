import streamlit as st
import torch
import whisper
import tempfile
from translate import Translator
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

    # Use Whisper's built-in decoding
    options = whisper.DecodingOptions(language=None, task="transcribe", without_timestamps=True)
    result = whisper.decode(model, mel, options)

    return result.text, result.language

def translate_to_welsh(text):
    translator = Translator(to_lang="cy")
    try:
        return translator.translate(text)
    except Exception as e:
        return f"Translation error: {e}"

def main():
    st.title("Whisper Audio Transcription: Default tiny whisper vs fine-tuned")

    st.subheader("Translate English to Welsh")

    english_input = st.text_area("Enter English Text", height=100)

    if st.button("Translate to Welsh"):
        if english_input.strip():
            welsh_translation = translate_to_welsh(english_input.strip())
            st.success("Welsh Translation:")
            st.text_area("Welsh", welsh_translation, height=100)
        else:
            st.warning("Please enter some English text.")

    st.markdown("---")

    st.write("Upload an audio file to compare transcriptions between the default and your fine-tuned Whisper model.")

    uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "flac", "ogg"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        default_model, fine_tuned_model, tokenizer, device = load_models()

        st.audio(uploaded_file, format=uploaded_file.type)
        st.write("Transcribing...")

        default_text, default_lang = transcribe_audio(default_model, device, tmp_file_path)
        tuned_text, tuned_lang = transcribe_audio(fine_tuned_model, device, tmp_file_path)

        st.success("Transcription complete!")

        st.subheader("Detected Language")
        st.write(f"Default model: `{default_lang}`")
        st.write(f"Fine-tuned model: `{tuned_lang}`")

        st.subheader("Default Whisper Transcription")
        st.text_area("Default", default_text, height=150)

        st.subheader("Fine-Tuned Whisper Transcription")
        st.text_area("Fine-Tuned", tuned_text, height=150)

if __name__ == "__main__":
    main()
