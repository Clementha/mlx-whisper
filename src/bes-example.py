import torch
import whisper
from torch.utils.data import Dataset, DataLoader
from utils import get_device 
from tqdm import tqdm

FILE_PATH = "audio/Clem--Bes.m4a"
EPOCHS = 3
LEARNING_RATE = 1e-5
BATCH_SIZE = 3

def whisper_without_fine_tuning(model, audio, device):
    options = whisper.DecodingOptions()
    log_mel = whisper.log_mel_spectrogram(audio).to(device)
    response = whisper.decode(model, log_mel, options)
    return response.text

# --- Set up Dataloaders for training and evaluation ---
class AudioDataset(Dataset):
    def __init__(self, file_path_list):
        self.file_paths = file_path_list

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        return audio

def train(model, tokenizer, train_dataloader):
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


    for epoch in range(EPOCHS):
        print(f"----Epoch {epoch + 1}/{EPOCHS}----")
        for batch_idx, batch in  enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
            audio = batch.squeeze(0) # This might mean that it only grabs the first 
            print("Whisper prediction without fine tuning: ", whisper_without_fine_tuning(model, audio, device))

            mel_unsqueezed = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(device)
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

if __name__ == "__main__":
    device = get_device()
    model = whisper.load_model("tiny", device=device)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

    train_dataset = AudioDataset(["audio/Clem--Bes.m4a", "audio/Helen--Bes.m4a", "audio/Ethan--Bes.m4a"])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train(model, tokenizer, train_dataloader)