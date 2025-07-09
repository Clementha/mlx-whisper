import streamlit as st
import torch
import whisper
import tempfile

@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("tiny", device=device)

    # Load your fine-tuned weights
    model.load_state_dict(torch.load("fine_tuned_whisper_tiny.pth", map_location=device))
    model.to(device)
    model.eval()

    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
    return model, tokenizer, device

def transcribe_audio(model, tokenizer, device, audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)

    ids = []
    ids += [tokenizer.sot]
    ids += [tokenizer.language_token]
    ids += [tokenizer.transcribe]
    ids += [tokenizer.no_timestamps]
    ids += [tokenizer.eot]
    tokens = torch.tensor(ids, device=device).unsqueeze(0)

    with torch.no_grad():
        prediction = model(tokens=tokens, mel=mel.unsqueeze(0))  # raw model output

    predicted_ids = prediction.argmax(dim=-1)[0].tolist()
    text = tokenizer.decode(predicted_ids)

    return prediction, text  # returning both

def main():
    st.title("Fine-tuned Whisper Audio Transcription")

    st.write("Upload an audio file to transcribe it using your fine-tuned Whisper model.")

    uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "flac", "ogg"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        model, tokenizer, device = load_model()

        st.audio(uploaded_file, format=uploaded_file.type)
        st.write("Transcribing...")

        raw_output, transcription = transcribe_audio(model, tokenizer, device, tmp_file_path)

        st.success("Transcription complete!")
        st.text_area("Transcription", transcription, height=150)

        st.subheader("Raw Whisper Output (tensor)")
        st.code(str(raw_output), language="python")

if __name__ == "__main__":
    main()
