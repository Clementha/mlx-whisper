import streamlit as st
import streamlit.components.v1 as components
import torch
import whisper
import tempfile
import subprocess
import os
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

def convert_uploaded_webm_to_wav(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_webm:
        tmp_webm.write(uploaded_file.read())
        tmp_webm_path = tmp_webm.name

    wav_path = tmp_webm_path.replace(".webm", ".wav")
    command = [
        "ffmpeg",
        "-y",
        "-i", tmp_webm_path,
        "-ar", "16000",
        "-ac", "1",
        wav_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return tmp_webm_path, wav_path

def transcribe_audio(model, device, audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)

    options = whisper.DecodingOptions(language=None, task="transcribe", without_timestamps=True)
    result = whisper.decode(model, mel, options)

    return result.text, result.language

def main():
    st.title("Whisper Audio Transcription")

    record_audio_js = """
    <script>
    let chunks = [];
    let mediaRecorder;
    let audioBlob;

    const startBtnId = 'startBtn';
    const stopBtnId = 'stopBtn';
    const clearBtnId = 'clearBtn';
    const downloadLinkId = 'downloadLink';
    const audioElementId = 'audioPlayback';

    function setButtonState(startEnabled, stopEnabled, clearEnabled, saveEnabled) {
        document.getElementById(startBtnId).disabled = !startEnabled;
        document.getElementById(stopBtnId).disabled = !stopEnabled;
        document.getElementById(clearBtnId).disabled = !clearEnabled;
        document.getElementById(downloadLinkId).style.display = saveEnabled ? 'inline' : 'none';
    }

    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            setButtonState(false, true, false, false);

            mediaRecorder.ondataavailable = e => {
                chunks.push(e.data);
            };

            mediaRecorder.onstop = e => {
                audioBlob = new Blob(chunks, { 'type' : 'audio/webm' });
                chunks = [];

                let audioURL = window.URL.createObjectURL(audioBlob);
                const audioElement = document.getElementById(audioElementId);
                audioElement.src = audioURL;

                const downloadLink = document.getElementById(downloadLinkId);
                downloadLink.href = audioURL;
                downloadLink.download = "recording.webm";

                setButtonState(true, false, true, true);
            };
        });
    }

    function stopRecording() {
        if(mediaRecorder && mediaRecorder.state === "recording"){
            mediaRecorder.stop();
            setButtonState(true, false, true, true);
        }
    }

    function clearRecording() {
        const audioElement = document.getElementById(audioElementId);
        audioElement.src = "";

        setButtonState(true, false, false, false);

        chunks = [];
        audioBlob = null;
    }

    window.onload = () => {
        setButtonState(true, false, false, false);
    };
    </script>

    <button id="startBtn" onclick="startRecording()">Start Recording</button>
    <button id="stopBtn" onclick="stopRecording()">Stop Recording</button>
    <button id="clearBtn" onclick="clearRecording()">Clear Recording</button>
    <a id="downloadLink" style="display: none; margin-left: 10px;" href="#" download>💾 Save Audio</a>

    <br><br>
    <audio id="audioPlayback" controls></audio>
    """

    components.html(record_audio_js, height=250)

    st.markdown("---")
    st.header("Upload your saved audio file (.webm)")

    uploaded_file = st.file_uploader("Upload the audio file you saved from above", type=["webm"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/webm")
        if st.button("🎤 Transcribe Uploaded Audio"):
            with st.spinner("Converting and transcribing..."):
                webm_path, wav_path = convert_uploaded_webm_to_wav(uploaded_file)

                default_model, fine_tuned_model, tokenizer, device = load_models()
                default_text, default_lang = transcribe_audio(default_model, device, wav_path)
                tuned_text, tuned_lang = transcribe_audio(fine_tuned_model, device, wav_path)

            st.success("Transcription complete!")

            st.subheader("Detected Language")
            st.write(f"Default model: `{default_lang}`")
            st.write(f"Fine-tuned model: `{tuned_lang}`")

            st.subheader("Default Whisper Transcription")
            st.text_area("Default", default_text, height=150)

            st.subheader("Fine-Tuned Whisper Transcription")
            st.text_area("Fine-Tuned", tuned_text, height=150)

            os.remove(webm_path)
            os.remove(wav_path)

if __name__ == "__main__":
    main()
