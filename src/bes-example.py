import torch
import whisper
from utils import get_device 

FILE_PATH = "src/Helen--Bes.m4a"
EPOCHS = 3
LEARNING_RATE = 1e-5

def train(model, audio, tokenizer):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    ids = []
    ids += [tokenizer.sot]
    ids += [tokenizer.language_token]
    ids += [tokenizer.transcribe]
    ids += tokenizer.encode(" Hello, my name is Bes.")
    ids += [tokenizer.eot]
    train_tokens = torch.tensor(ids, device=device).unsqueeze(0)

    mel_unsqueezed = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(device)

    for epoch in range(EPOCHS):
        print(f"----Epoch {epoch + 1}/{EPOCHS}----")

        prediction = model(tokens=train_tokens, mel=mel_unsqueezed)
        target = train_tokens[:, 1:].contiguous()  # Skip the start token

        loss = criterion(prediction.transpose(1, 2), train_tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = model(tokens=train_tokens, mel=mel_unsqueezed)
        prediction = prediction[:, :-1, :].contiguous()  # Remove the last token

        print("Loss: ", loss.item())
        # print("Ids Target: ", target.squeeze().tolist())
        # print("Ids Pred: ", torch.argmax(prediction, dim=-1).squeeze().tolist())
        print("Text target: ", tokenizer.decode(target.squeeze().tolist()))
        print("Text pred: ", tokenizer.decode(torch.argmax(prediction, dim=-1).squeeze().tolist()))

def whisper_without_fine_tuning(model, audio, device):
    options = whisper.DecodingOptions()
    log_mel = whisper.log_mel_spectrogram(audio).to(device)
    response = whisper.decode(model, log_mel, options)
    return response.text

if __name__ == "__main__":
    device = get_device()
    model = whisper.load_model("tiny", device=device)

    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

    audio = whisper.load_audio(FILE_PATH)
    audio = whisper.pad_or_trim(audio)
    
    print("Whisper prediction without fine tuning: ", whisper_without_fine_tuning(model, audio, device))

    train(model, audio, tokenizer)