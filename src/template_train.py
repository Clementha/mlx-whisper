import torch
import whisper
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import get_device, init_wandb
from template_train_utils import log_without_fine_tuning, log_predict_targets, compute_avg_masked_accuracy_per_batch, average_whisper_accuracy_before_ft, gen_token_ids_with_special_tokens

EPOCHS = 5
LEARNING_RATE = 1e-5
BATCH_SIZE = 3

class AudioDataset(Dataset):
    def __init__(self, file_path_list):
        self.file_paths = file_path_list

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        return audio


def evaluate(model, tokenizer, eval_dataloader, batch_idx, wandb_pre_fine_tune_logs):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0
    ids = gen_token_ids_with_special_tokens(tokenizer, " Hello, my name is Bes.")
    eval_text_table = wandb.Table(columns=["sample_num", "pre_fine_tuning", "last_predicted", "target", "last_predicted_tokens", "target_tokens"])

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            audio_batch = batch.to(device)

            mel = whisper.log_mel_spectrogram(audio_batch).to(device)  # [B, 80, T]

            B = mel.size(0)
            eval_tokens = torch.tensor(ids, device=device).unsqueeze(0).repeat(B, 1)  # [B, T]
            target = eval_tokens[:, 1:].contiguous()  # [B, T-1]

            prediction = model(tokens=eval_tokens, mel=mel)  # [B, T, Vocab]

            if batch_idx == 0:
                log_predict_targets(eval_text_table, tokenizer, wandb_pre_fine_tune_logs, target, prediction, 1)

            loss = criterion(prediction[:, :-1, :].transpose(1, 2), target)
            total_loss += loss.item()

            batch_accuracy = compute_avg_masked_accuracy_per_batch(prediction, target, B)
            total_accuracy += batch_accuracy

            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_accuracy = total_accuracy / total_batches if total_batches > 0 else 0
    print(f"Evaluation - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

    return avg_loss, avg_accuracy, eval_text_table


def train(model, tokenizer, train_dataloader, eval_dataloader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    ids = gen_token_ids_with_special_tokens(tokenizer, " Hello, my name is Bes.")

    wandb_pre_fine_tune_logs = []
    for epoch in range(EPOCHS):
        text_table = wandb.Table(columns=["sample_num", "pre_fine_tuning", "last_predicted", "target", "last_predicted_tokens", "target_tokens"])
        print(f"\n---- Epoch {epoch + 1}/{EPOCHS} ----")
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            audio_batch = batch.to(device)  # [B, N]

            mel = whisper.log_mel_spectrogram(audio_batch).to(device)  # [B, 80, T]

            B = mel.size(0)
            train_tokens_batch = torch.tensor(ids, device=device).unsqueeze(0).repeat(B, 1)
            target = train_tokens_batch[:, 1:].contiguous()  # [B, T-1]
            
            if epoch == 0 and batch_idx == 0:
                log_without_fine_tuning(model, audio_batch, wandb_pre_fine_tune_logs)
                avg_whisper_accuracy = average_whisper_accuracy_before_ft(model, audio_batch, target, tokenizer)
            
            prediction = model(tokens=train_tokens_batch, mel=mel)  # [B, T, Vocab]
            loss = criterion(prediction[:, :-1, :].transpose(1, 2), target)    # [B, V, T] vs [B, T]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_batch_accuracy = compute_avg_masked_accuracy_per_batch(prediction, target, B)
            
            if epoch == EPOCHS - 1 and batch_idx == 0:
                log_predict_targets(text_table, tokenizer, wandb_pre_fine_tune_logs, target, prediction, B)
            
            eval_avg_loss, eval_avg_accuracy, eval_text_table = evaluate(model, tokenizer, eval_dataloader, batch_idx, wandb_pre_fine_tune_logs)
            wandb.log({"epoch": epoch + 1, "loss": loss.item(), "training_text": text_table, "avg_batch_accuracy": avg_batch_accuracy, "avg_whisper_accuracy": avg_whisper_accuracy, "eval_avg_loss": eval_avg_loss, "eval_avg_accuracy": eval_avg_accuracy, "eval_text_table": eval_text_table })


if __name__ == "__main__":
    init_wandb()
    device = get_device()
    model = whisper.load_model("tiny", device=device)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

    train_dataset = AudioDataset(["audio/Clem--Bes.m4a", "audio/Helen--Bes.m4a"])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    eval_dataset = AudioDataset(["audio/Ethan--Bes.m4a"])
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train(model, tokenizer, train_dataloader, eval_dataloader)
    torch.save(model.state_dict(), "fine_tuned_whisper_tiny.pth")

    wandb.finish()