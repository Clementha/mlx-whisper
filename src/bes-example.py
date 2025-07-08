import torch
import whisper
from utils import get_device 

FILE_PATH = "src/Helen--Bes.m4a"

device = get_device()
model = whisper.load_model("tiny", device=device)

# Load and process the audio file
audio = whisper.load_audio(FILE_PATH)
audio = whisper.pad_or_trim(audio)
log_mel = whisper.log_mel_spectrogram(audio).to(device)
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

options = whisper.DecodingOptions()
response = whisper.decode(model, log_mel, options)
print("whisper prediction without fine tuning: ", response.text)

# Preparing input for target for the model to train on and learn
ids = []
ids += [tokenizer.sot]
ids += [tokenizer.language_token]
ids += [tokenizer.transcribe]
ids += tokenizer.encode(" Hello, my name is Bes.")
ids += [tokenizer.eot]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()
model.train()

train_tokens = torch.tensor(ids, device=device).unsqueeze(0)
mel_unsqueezed = log_mel.unsqueeze(0).to(device)
prediction = model(tokens=train_tokens, mel=mel_unsqueezed)
target = train_tokens[:, 1:].contiguous()  # Skip the start token

print("--- Before training ---")
print("Ids Target: ", target.squeeze().tolist())
print("Ids Pred: ", torch.argmax(prediction, dim=-1).squeeze().tolist())
print("Text target: ", tokenizer.decode(target.squeeze().tolist()))
print("Text pred: ", tokenizer.decode(torch.argmax(prediction, dim=-1).squeeze().tolist()))

# Training the model
loss = criterion(prediction.transpose(1, 2), train_tokens)
print("Loss: ", loss.item())
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("--- After training ---")
model.eval()
prediction = model(tokens=train_tokens, mel=mel_unsqueezed)
prediction = prediction[:, :-1, :].contiguous()  # Remove the last token

print("Ids Target: ", target.squeeze().tolist())
print("Ids Pred: ", torch.argmax(prediction, dim=-1).squeeze().tolist())
print("Text target: ", tokenizer.decode(target.squeeze().tolist()))
print("Text pred: ", tokenizer.decode(torch.argmax(prediction, dim=-1).squeeze().tolist()))

loss = criterion(prediction.transpose(1, 2), target)
print("Loss: ", loss.item())