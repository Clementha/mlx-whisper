import torch
import whisper
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import get_device, init_wandb
from train_utils import log_without_fine_tuning, log_predict_targets

FILE_PATH = "audio/Clem--Bes.m4a"
EPOCHS = 2
LEARNING_RATE = 1e-5
BATCH_SIZE = 3

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
    ids += [tokenizer.no_timestamps]
    ids += tokenizer.encode(" Hello, my name is Bes.")
    ids += [tokenizer.eot]
    train_tokens = torch.tensor(ids, device=device).unsqueeze(0)  # [1, T]

    wandb_pre_fine_tune_logs = []
    for epoch in range(EPOCHS):
        text_table = wandb.Table(columns=["sample_num", "pre_fine_tuning", "last_predicted", "target", "last_predicted_tokens", "target_tokens"])
        print(f"\n---- Epoch {epoch + 1}/{EPOCHS} ----")
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            audio_batch = batch.to(device)  # [B, N]
            if epoch == 0 and batch_idx == 0:
                log_without_fine_tuning(model, audio_batch, wandb_pre_fine_tune_logs)

            mel = whisper.log_mel_spectrogram(audio_batch).to(device)  # [B, 80, T]

            B = mel.size(0)
            train_tokens_batch = train_tokens.repeat(B, 1)  # [B, T]
            target = train_tokens_batch[:, 1:].contiguous()  # [B, T-1]

            prediction = model(tokens=train_tokens_batch, mel=mel)  # [B, T, Vocab]
            loss = criterion(prediction[:, :-1, :].transpose(1, 2), target)    # [B, V, T] vs [B, T]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item():.4f}")
            if epoch == EPOCHS - 1 and batch_idx == 0:
                log_predict_targets(text_table, tokenizer, wandb_pre_fine_tune_logs, target, prediction)
            wandb.log({"epoch": epoch + 1, "loss": loss.item(), f"text": text_table})

if __name__ == "__main__":
    init_wandb()
    device = get_device()
    model = whisper.load_model("tiny", device=device)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

    train_dataset = AudioDataset(["audio/Clem--Bes.m4a", "audio/Helen--Bes.m4a", "audio/Ethan--Bes.m4a"])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train(model, tokenizer, train_dataloader)
    wandb.finish()