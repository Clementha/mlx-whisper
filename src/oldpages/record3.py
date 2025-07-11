import streamlit as st
import torch
import whisper
import tempfile
import numpy as np
from pydub import AudioSegment
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils import get_device, load_artifact_path, init_wandb

MODEL_VERSION = "v0"

@st.cache_resource(show_spinner=False)
def load_models():
    init_wandb()
    device = get_device()
    model_file_path = load_artifact_path("fine_tuned_welsh_model", MODEL_VERSION, "pth")

    default_model = whisper.load_model("tiny", device=device)
    default_model.to(device)
    default_model.eval()

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

def save_audio(frames, sample_rate=48000):
    if len(frames) == 0:
        return None
    audio_data = np.concatenate(frames, axis=0)
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=1,
    )
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_segment.export(tmp_file.name, format="wav")
    return tmp_file.name

def main():
    st.title("Whisper Audio Transcription (Record or Upload)")

    if "audio_frames" not in st.session_state:
        st.session_state.audio_frames = []

    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None

    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": []},  # Disable STUN servers to avoid errors
        async_processing=True,
    )

    if webrtc_ctx.audio_receiver:
        frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        if frames:
            st.session_state.audio_frames.extend([frame.to_ndarray(format="s16") for frame in frames])

    if st.session_state.audio_frames:
        if st.session_state.audio_path is None:
            st.session_state.audio_path = save_audio(st.session_state.audio_frames)

        st.audio(st.session_state.audio_path)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Transcribe Recorded Audio"):
                with st.spinner("Transcribing..."):
                    default_model, fine_tuned_model, tokenizer, device = load_models()

                    default_text, default_lang = transcribe_audio(default_model, device, st.session_state.audio_path)
                    tuned_text, tuned_lang = transcribe_audio(fine_tuned_model, device, st.session_state.audio_path)

                st.success("Transcription complete!")

                st.subheader("Detected Language")
                st.write(f"Default model: `{default_lang}`")
                st.write(f"Fine-tuned model: `{tuned_lang}`")

                st.subheader("Default Whisper Transcription")
                st.text_area("Default", default_text, height=150)

                st.subheader("Fine-Tuned Whisper Transcription")
                st.text_area("Fine-Tuned", tuned_text, height=150)

        with col2:
            if st.button("Clear Recording"):
                st.session_state.audio_frames = []
                st.session_state.audio_path = None
                st.experimental_rerun()

    else:
        st.info("Start speaking to record audio. When done, use 'Transcribe Recorded Audio' to process it.")

    uploaded_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_path = tmp_file.name

        st.audio(uploaded_file, format=uploaded_file.type)
        with st.spinner("Transcribing uploaded audio..."):
            default_model, fine_tuned_model, tokenizer, device = load_models()

            default_text, default_lang = transcribe_audio(default_model, device, audio_path)
            tuned_text, tuned_lang = transcribe_audio(fine_tuned_model, device, audio_path)

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
