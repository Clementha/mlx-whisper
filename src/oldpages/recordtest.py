import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import tempfile
import os
import soundfile as sf
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
    st.title("Audio Recording and Transcription")
    st.write("Record audio or upload a file, then transcribe it using Whisper models.")

    # Initialize session state
    if "audio_frames" not in st.session_state:
        st.session_state.audio_frames = []
    if "audio_file_path" not in st.session_state:
        st.session_state.audio_file_path = None
    if "transcription_complete" not in st.session_state:
        st.session_state.transcription_complete = False

    # File uploader fallback
    uploaded_file = st.file_uploader("Upload an audio file (wav, mp3, m4a, flac, ogg)", type=["wav", "mp3", "m4a", "flac", "ogg"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile.write(uploaded_file.read())
            st.session_state.audio_file_path = tmpfile.name
        st.success("Audio file uploaded!")

    st.markdown("---")
    st.write("**Or record audio below (works only in supported environments):**")

    def audio_frame_callback(frame):
        audio = frame.to_ndarray(format="flt32")
        st.session_state.audio_frames.append(audio)
        return av.AudioFrame.from_ndarray(audio, format="flt32", layout="mono")

    # WebRTC streamer for recording
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDRECV,
        audio_frame_callback=audio_frame_callback,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    # Recording controls
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Clear Recording"):
            st.session_state.audio_frames.clear()
            st.session_state.audio_file_path = None
            st.session_state.transcription_complete = False
            st.success("Cleared audio recording")
            st.rerun()

    with col2:
        if st.button("Save Recording"):
            if not st.session_state.audio_frames:
                st.warning("No audio recorded yet")
            else:
                audio_data = np.concatenate(st.session_state.audio_frames, axis=0)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    sf.write(tmpfile.name, audio_data, samplerate=48000)
                    st.session_state.audio_file_path = tmpfile.name
                    st.success("Recording saved!")
                    st.rerun()

    with col3:
        st.write("Use the widget above to start/stop recording.")

    # Show saved audio if available
    if st.session_state.audio_file_path and os.path.exists(st.session_state.audio_file_path):
        st.subheader("Audio Ready for Transcription")
        st.audio(st.session_state.audio_file_path, format="audio/wav")
        
        # Transcribe button
        if st.button("Transcribe Audio", type="primary"):
            if st.session_state.audio_file_path:
                with st.spinner("Loading models and transcribing..."):
                    default_model, fine_tuned_model, tokenizer, device = load_models()
                    default_text, default_lang = transcribe_audio(default_model, device, st.session_state.audio_file_path)
                    tuned_text, tuned_lang = transcribe_audio(fine_tuned_model, device, st.session_state.audio_file_path)
                
                st.session_state.transcription_complete = True
                st.session_state.default_text = default_text
                st.session_state.tuned_text = tuned_text
                st.session_state.default_lang = default_lang
                st.session_state.tuned_lang = tuned_lang
                
                st.success("Transcription complete!")
                st.rerun()

    # Show transcription results if available
    if st.session_state.get('transcription_complete', False):
        st.subheader("Transcription Results")
        
        st.write("**Detected Language:**")
        st.write(f"Default model: `{st.session_state.default_lang}`")
        st.write(f"Fine-tuned model: `{st.session_state.tuned_lang}`")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Default Whisper Transcription")
            st.text_area("Default", st.session_state.default_text, height=150, key="default_text_area")
        
        with col2:
            st.subheader("Fine-Tuned Whisper Transcription")
            st.text_area("Fine-Tuned", st.session_state.tuned_text, height=150, key="tuned_text_area")

if __name__ == "__main__":
    main() 