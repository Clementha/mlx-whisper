import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
import tempfile
import os
import soundfile as sf

st.title("Audio Recorder")

if "audio_frames" not in st.session_state:
    st.session_state.audio_frames = []

def audio_frame_callback(frame):
    audio = frame.to_ndarray(format="flt32")
    st.session_state.audio_frames.append(audio)
    return av.AudioFrame.from_ndarray(audio, format="flt32", layout="mono")

webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    mode=WebRtcMode.SENDRECV,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Clear"):
        st.session_state.audio_frames.clear()
        st.success("Cleared audio recording")

with col2:
    if st.button("Playback"):
        if not st.session_state.audio_frames:
            st.warning("No audio recorded yet")
        else:
            audio_data = np.concatenate(st.session_state.audio_frames, axis=0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                sf.write(tmpfile.name, audio_data, samplerate=48000)
                st.audio(tmpfile.name)
                os.unlink(tmpfile.name)

with col3:
    st.write("Use the widget above to start/stop recording.")
