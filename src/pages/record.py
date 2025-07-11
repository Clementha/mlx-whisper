import streamlit as st
import streamlit.components.v1 as components

st.title("Audio Recorder with Start, Stop, and Playback")

# Define a simple HTML + JS recorder
record_audio = """
<script>
let chunks = [];
let mediaRecorder;
let audioBlob;

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();

        mediaRecorder.ondataavailable = e => {
            chunks.push(e.data);
        };

        mediaRecorder.onstop = e => {
            audioBlob = new Blob(chunks, { 'type' : 'audio/wav; codecs=opus' });
            chunks = [];

            let audioURL = window.URL.createObjectURL(audioBlob);
            const audioElement = document.getElementById('audioPlayback');
            audioElement.src = audioURL;

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
    mediaRecorder.stop();
}

</script>

<button onclick="startRecording()">Start Recording</button>
<button onclick="stopRecording()">Stop Recording</button>

<br><br>
<audio id="audioPlayback" controls></audio>
"""

# Embed the recorder and receive messages
components.html(record_audio, height=200)

# Capture audio from the frontend
audio_data = st.query_params.get("audio")
# Use st.experimental_set_query_params to capture audio from JS is tricky,
# so instead, listen for postMessages with st.components.v1 — but this requires a workaround.
# For now, just show a note:
st.write("After you stop recording, playback appears above.")

