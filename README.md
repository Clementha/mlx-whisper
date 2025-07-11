# mlx-whisper
Use whisper model to process audio to do:
- Task 1. Template - Hello, my name is Bes. This will take three recordings of "Hello, my name is Bes." and fine tune whisper to be able to transcribe Bes (by default it is Bez/Beth etc)
- Task 2. Teach whisper tiny welsh! 

## Pre-requisities 
- Python 3.10
- uv - install via https://github.com/astral-sh/uv
- wandb log in details and be added to the project - https://wandb.ai/site or in remote terminal 
- `brew install ffmpeg` (you need this for whisper to work)
- `.env` - see example and fill it in. Entity is audio-party and we use mlx-whisper for the template and mlx-whisper-welsh for welsh

## Brainstorming/Architecture
https://excalidraw.com/#room=74a7e13165b90cd7eadd,S097Zfcno-P6c-HlbWnUPg

<img width="1046" height="628" alt="Screenshot 2025-07-10 at 17 14 22" src="https://github.com/user-attachments/assets/d1ebdf5e-25e0-4439-a705-fff4f6115916" />

## Task 1. Template - Hello, my name is Bes
0. You'll need to populate .env (use .env.example as an example. For this Hello my name is Bes - the entity is audio-party and project is mlx-whisper). Also add a folder called 'audio' and put the three audio clippings inside. 
1. To train: `uv run template_train.py` that will train on audio clippings of "Hello, my name is Bes". See bes-original.ipynb that is the original code adapted. It calculates accuracy removing the special tokens and seeing how the words match.
2. To run inference: `uv run streamlit run src/template_inference.py`

## Task 2. Training whisper tiny welsh
See all code with welsh_... prefix
- Downloads the dataset: https://huggingface.co/datasets/EthanGLEdwards/welsh-transcription-samples/viewer/default/train?views%5B%5D=train  - this is a mini dataset for coding etc, not used trained on (see `making_dataset.ipynb` that generates it and `welsh_dataset.ipynb` to read it and view)
1. Dataset created - see `welsh_3split.py` that generates the dataset, and `welsh_dataset_push_to_hf.py` that pushes to hugging face. Basis is: https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0 & output is: https://huggingface.co/datasets/ClemSummer/welsh-transcription-samples-7k 
2. Run `welsh_train.py` that will train and evaludate the model
3. To run inference: `uv run streamlit run src/welsh_audio_inference.py`

Image of it working!

<img width="1495" height="668" alt="image" src="https://github.com/user-attachments/assets/f190c85c-ef23-4deb-ae72-ae5147176d4d" />

Weights and biases:

<img width="1324" height="644" alt="Screenshot 2025-07-10 at 17 20 11" src="https://github.com/user-attachments/assets/7bd00ac4-304b-4f56-b532-21a52aeeb1bb" />

