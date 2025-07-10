# mlx-whisper
Use whisper model to process audio to do:
- Task 1. Template - Hello, my name is Bes. This will take three recordings of "Hello, my name is Bes." and fine tune whisper to be able to transcribe Bes (by default it is Bez/Beth etc)
- Task 2. Teach whisper tiny welsh! 

## Pre-requisities 
- Python 3.10
- uv - install via https://github.com/astral-sh/uv
- wandb log in details and be added to the project - https://wandb.ai/site or in remote terminal 
- `brew install ffmpeg` (you need this for whisper to work)
- `.env` - see example

## Brainstorming
https://excalidraw.com/#room=74a7e13165b90cd7eadd,S097Zfcno-P6c-HlbWnUPg

## Task 1. Template - Hello, my name is Bes
0. You'll need to populate .env (use .env.example as an example. For this Hello my name is Bes - the entity is audio-party and project is mlx-whisper). Also add a folder called 'audio' and put the three audio clippings inside. 
1. To train: `uv run template_train.py` that will train on audio clippings of "Hello, my name is Bes". See bes-original.ipynb that is the original code adapted. It calculates accuracy removing the special tokens and seeing how the words match.
2. To run inference: `uv run streamlit run src/template_inference.py`

## Task 2. Training whisper tiny welsh
See all code with welsh_... prefix
- Downloads the dataset: https://huggingface.co/datasets/EthanGLEdwards/welsh-transcription-samples/viewer/default/train?views%5B%5D=train  - this is a mini dataset for coding etc, not used trained on (see `making_dataset.ipynb` that generates it and `welsh_dataset.ipynb` to read it and view)
1. Dataset created - see `welsh_3split.py` that generates the dataset. Basis is: https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0 & output is: https://huggingface.co/datasets/ClemSummer/welsh-transcription-samples-7k 
2. 