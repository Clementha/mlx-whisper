import streamlit as st
import streamlit.components.v1 as components

st.title("Audio Recorder with Start, Stop, and Playback")

record_audio = """
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

        setButtonState(false, true, false);  // Start disabled, Stop enabled, Clear disabled

        mediaRecorder.ondataavailable = e => {
            chunks.push(e.data);
        };

        mediaRecorder.onstop = e => {
            audioBlob = new Blob(chunks, { 'type' : 'audio/wav; codecs=opus' });
            chunks = [];

            let audioURL = window.URL.createObjectURL(audioBlob);
            const audioElement = document.getElementById(audioElementId);
            audioElement.src = audioURL;

            // Enable Clear button now that we have a recording
            setButtonState(true, false, true);

            // send base64 audio data back to Streamlit
            var reader = new FileReader();
            reader.readAsDataURL(audioBlob); 
            reader.onloadend = function() {
                const base64data = reader.result;
                window.parent.postMessage({ audio: base64data }, "*");
            }
        };
    });
}

function stopRecording() {
    if(mediaRecorder && mediaRecorder.state === "recording"){
        mediaRecorder.stop();
        setButtonState(true, false, true);  // After stopping, start enabled, stop disabled, clear enabled
    }
}

function clearRecording() {
    // Reset audio playback
    const audioElement = document.getElementById(audioElementId);
    audioElement.src = "";

    // Reset buttons
    setButtonState(true, false, false);

    // Clear any stored audioBlob or chunks
    chunks = [];
    audioBlob = null;

    // Inform Streamlit to clear audio (optional)
    window.parent.postMessage({ audio: null }, "*");
}

window.onload = () => {
    setButtonState(true, false, false);  // Initially: start enabled, stop and clear disabled
};
</script>

<button id="startBtn" onclick="startRecording()">Start Recording</button>
<button id="stopBtn" onclick="stopRecording()">Stop Recording</button>
<button id="clearBtn" onclick="clearRecording()">Clear Recording</button>

<br><br>
<audio id="audioPlayback" controls></audio>
"""

components.html(record_audio, height=230)

st.write("After you stop recording, playback appears above.")
