import torch
import whisper
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import get_device, init_wandb

FILE_PATH = "audio/Clem--Bes.m4a"
EPOCHS = 3
LEARNING_RATE = 1e-5
BATCH_SIZE = 3

def whisper_without_fine_tuning(model, audio_batch, device):
    options = whisper.DecodingOptions()
    outputs = []
    for audio in audio_batch:
        log_mel = whisper.log_mel_spectrogram(audio).to(device)
        response = whisper.decode(model, log_mel, options)
        outputs.append(response.text)
    return outputs

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

    # Prepare target token sequence (e.g., for dummy training target)
    ids = []
    ids += [tokenizer.sot]
    ids += [tokenizer.language_token]
    ids += [tokenizer.transcribe]
    ids += [tokenizer.no_timestamps]
    ids += tokenizer.encode(" Hello, my name is Bes.")
    ids += [tokenizer.eot]
    train_tokens = torch.tensor(ids, device=device).unsqueeze(0)  # [1, T]

    for epoch in range(EPOCHS):
        text_table = wandb.Table(columns=["sample_num", "pre_fine_tuning", "predicted", "target"])
        print(f"\n---- Epoch {epoch + 1}/{EPOCHS} ----")
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            audio_batch = batch.to(device)  # [B, N]
            # Show whisper prediction without fine-tuning
            print("Whisper predictions (no fine-tuning):")
            predictions_raw = whisper_without_fine_tuning(model, audio_batch, device)
            wandb_logs = {
                "sample_num": [],
                "pre_fine_tuning": [],
            }
            for i, text in enumerate(predictions_raw):
                wandb_logs["pre_fine_tuning"].append(text)
                wandb_logs["sample_num"].append(f"Sample {i + 1}")
                print(f"  Sample {i + 1}: {text}")

            # Compute log-mel spectrograms for batch
            mel = whisper.log_mel_spectrogram(audio_batch).to(device)  # [B, 80, T]

            # Repeat the token sequence for each item in the batch
            B = mel.size(0)
            train_tokens_batch = train_tokens.repeat(B, 1)  # [B, T]
            target = train_tokens_batch[:, 1:].contiguous()  # [B, T-1]

            # Forward pass
            prediction = model(tokens=train_tokens_batch, mel=mel)  # [B, T, Vocab]
            # Align prediction and target sequence lengths by slicing prediction
            loss = criterion(prediction[:, :-1, :].transpose(1, 2), target)    # [B, V, T] vs [B, T]

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Predictions for logging
            pred_tokens = torch.argmax(prediction[:, :-1, :], dim=-1).contiguous()  # [B, T-1]
            print(f"Loss: {loss.item():.4f}")
            for i in range(B):
                print(f"Sample {i + 1}:")
                target_text = tokenizer.decode(target[i].tolist())
                predicted_text = tokenizer.decode(pred_tokens[i].tolist())
                text_table.add_data(
                    wandb_logs["sample_num"][i],
                    wandb_logs["pre_fine_tuning"][i],
                    predicted_text,
                    target_text
                )
                print("  Target text:   ", target_text)
                print("  Predicted text:", predicted_text)
            wandb.log({"epoch": epoch + 1, "loss": loss.item(), f"text_epoch_{epoch+1}": text_table})

if __name__ == "__main__":
    init_wandb()
    device = get_device()
    model = whisper.load_model("tiny", device=device)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

    train_dataset = AudioDataset(["audio/Clem--Bes.m4a", "audio/Helen--Bes.m4a", "audio/Ethan--Bes.m4a"])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train(model, tokenizer, train_dataloader)
    wandb.finish()