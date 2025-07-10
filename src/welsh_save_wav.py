import sys
from gtts import gTTS
from pydub import AudioSegment

def main():
    if len(sys.argv) < 2:
        print("Usage: python save_welsh_tts.py \"Your text here\"")
        sys.exit(1)

    text = sys.argv[1]
    lang = "cy"  # Welsh language code

    # Generate TTS
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    print("✅ Saved MP3 as output.mp3")

    # Convert MP3 to WAV
    sound = AudioSegment.from_mp3("output.mp3")
    sound.export("output.wav", format="wav")
    print("✅ Saved WAV as output.wav")

if __name__ == "__main__":
    main()
