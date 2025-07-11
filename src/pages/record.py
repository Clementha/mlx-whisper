import streamlit as st
import streamlit.components.v1 as components
import torch
import whisper
import tempfile
import base64
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

def save_audio_from_base64(base64_audio):
    header, encoded = base64_audio.split(",", 1)
    data = base64.b64decode(encoded)
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_file.write(data)
    tmp_file.close()
    return tmp_file.name

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
    const audioElementId = 'audioPlayback';

    function setButtonState(startEnabled, stopEnabled, clearEnabled) {
        document.getElementById(startBtnId).disabled = !startEnabled;
        document.getElementById(stopBtnId).disabled = !stopEnabled;
        document.getElementById(clearBtnId).disabled = !clearEnabled;
    }

    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            setButtonState(false, true, false);

            mediaRecorder.ondataavailable = e => {
                chunks.push(e.data);
            };

            mediaRecorder.onstop = e => {
                audioBlob = new Blob(chunks, { 'type' : 'audio/wav; codecs=opus' });
                chunks = [];

                let audioURL = window.URL.createObjectURL(audioBlob);
                const audioElement = document.getElementById(audioElementId);
                audioElement.src = audioURL;

                setButtonState(true, false, true);

                var reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = function() {
                    const base64data = reader.result;

                    // Find hidden input by custom attribute and set base64
                    const input = window.parent.document.querySelector('input[data-base64-input]');
                    if(input){
                        input.value = base64data;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                }
            };
        });
    }

    function stopRecording() {
        if(mediaRecorder && mediaRecorder.state === "recording"){
            mediaRecorder.stop();
            setButtonState(true, false, true);
        }
    }

    function clearRecording() {
        const audioElement = document.getElementById(audioElementId);
        audioElement.src = "";

        setButtonState(true, false, false);

        chunks = [];
        audioBlob = null;

        const input = window.parent.document.querySelector('input[data-base64-input]');
        if(input){
            input.value = "";
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }

    window.onload = () => {
        setButtonState(true, false, false);
    };
    </script>

    <button id="startBtn" onclick="startRecording()">Start Recording</button>
    <button id="stopBtn" onclick="stopRecording()">Stop Recording</button>
    <button id="clearBtn" onclick="clearRecording()">Clear Recording</button>

    <br><br>
    <audio id="audioPlayback" controls></audio>
    """

    components.html(record_audio_js, height=230)

    # Hidden input to receive base64 audio from JS
    audio_base64 = st.text_input(
        "",
        value="",
        label_visibility="collapsed",
        key="audio_base64",
        help="hidden base64 audio input",
        args={"data-base64-input": True}  # custom attribute for JS selector
    )

    # Hide the input visually via CSS
    st.markdown(
        """
        <style>
        input[data-base64-input] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # If audio recorded, show audio player
    if audio_base64:
        st.audio(audio_base64, format="audio/wav")

        # Auto transcribe once audio_base64 is available
        audio_path = save_audio_from_base64(audio_base64)

        with st.spinner("Loading models and transcribing..."):
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
