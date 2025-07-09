from datasets import load_dataset

dataset = load_dataset("EthanGLEdwards/welsh-transcription-samples")

# Access the training split
train_data = dataset['train']

# Example: First entry
transcription = train_data[0]['caption']

print(f"Transcription: {transcription}")

