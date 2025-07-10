from datasets import load_dataset
import torch
import whisper
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import get_device, init_wandb
from welsh_train_utils import log_predict_targets, compute_avg_masked_accuracy_per_batch, gen_token_ids_with_special_tokens
import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence

EPOCHS = 5
LEARNING_RATE = 1e-5
BATCH_SIZE = 3

class AudioDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        caption = self.dataset[idx]['caption']
        id = self.dataset[idx]['sample_id']  # e.g. "welsh_001"
        language = self.dataset[idx]['language'] # cy or en
        caption_ids = gen_token_ids_with_special_tokens(self.tokenizer, language, caption)
        mel = preprocess_audio(self.dataset[idx])
        return {
            "mel": mel,
            "caption_ids": caption_ids,
            "sample_id": id
        }

def whisper_collate_fn(batch):
    mels = [item["mel"] for item in batch]  # Already log-mel spectrogram
    caption_ids = [item["caption_ids"] for item in batch]  # Already tokenized
    sample_ids = [item["sample_id"] for item in batch]  # Keep sample IDs
    mels = torch.stack(mels)
    caption_ids = pad_sequence(caption_ids, batch_first=True, padding_value=caption_ids[0][-1].item())  # Use EOT as pad

    return {
        "mel": mels,
        "caption_ids": caption_ids,
        "sample_id": sample_ids  # Keep sample IDs
    }
exp_sr = 16000  # Whisper expects 16kHz audio
def preprocess_audio(example):
    audio = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]

    waveform = torch.tensor(audio, dtype=torch.float64)  # 👈 force Double here

    if sr != exp_sr:
        waveform = waveform.unsqueeze(0)  # (1, samples)

        # 👇 set dtype so kernel matches waveform
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=exp_sr,
            dtype=waveform.dtype
        )
        waveform = resampler(waveform).squeeze(0)

    audio = waveform.to(torch.float32).numpy()  # convert back to float32 for Whisper
    audio = whisper.pad_or_trim(audio)
    return whisper.log_mel_spectrogram(audio)

def evaluate(model, tokenizer, eval_dataloader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0

    val_samples = {}

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            audio_batch = batch["mel"].to(device)
            eval_tokens = batch["caption_ids"].to(device)
            sample_ids = batch["sample_id"]

            B = audio_batch.size(0)
            target = eval_tokens[:, 1:].contiguous()

            prediction = model(tokens=eval_tokens, mel=audio_batch)

            loss = criterion(prediction[:, :-1, :].transpose(1, 2), target)
            total_loss += loss.item()

            batch_accuracy = compute_avg_masked_accuracy_per_batch(prediction, target, B)
            total_accuracy += batch_accuracy
            total_batches += 1

            for i in range(B):
                pred_text, target_text, pred_tokens, target_tokens = log_predict_targets(tokenizer, target[i], prediction[i])
                val_samples[sample_ids[i]] = {
                    "pred_text": pred_text,
                    "pred_tokens": f"{pred_tokens}",
                    "target_text": target_text,
                    "target_tokens": f"{target_tokens}"
                }

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_accuracy = total_accuracy / total_batches if total_batches > 0 else 0

    validation_table = wandb.Table(columns=["sample_id", "predicted_text", "predicted_tokens", "target", "target_tokens"])
    for sample_id, data in val_samples.items():
        validation_table.add_data(
            sample_id,
            data.get("pred_text", ""),
            data.get("pred_tokens", ""),
            data.get("target_text", ""),
            data.get("target_tokens", "")
        )
    wandb.log({"validation_text": validation_table})

    return avg_loss, avg_accuracy

def train(model, tokenizer, train_dataloader, eval_dataloader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    samples = {}

    for epoch in range(EPOCHS):
        print(f"=== Epoch {epoch+1} ===")
        running_loss = 0.0
        running_accuracy = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            mels = batch["mel"].to(device)
            caption_ids = batch["caption_ids"].to(device)
            B = mels.size(0)

            target = caption_ids[:, 1:].contiguous()  # [B, T-1]
            prediction = model(tokens=caption_ids, mel=mels)  # [B, T, Vocab]

            avg_batch_accuracy = compute_avg_masked_accuracy_per_batch(prediction, target, B)
            loss = criterion(prediction[:, :-1, :].transpose(1, 2), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += avg_batch_accuracy
            total_batches += 1

            if epoch == 0:
                for i in range(B):
                    first_pred_text, _, first_pred_tokens, _ = log_predict_targets(tokenizer, target[i], prediction[i])
                    samples[batch["sample_id"][i]] = {
                        "first_pred_text": first_pred_text,
                        "first_pred_tokens": f"{first_pred_tokens}"
                    }

            if epoch == EPOCHS - 1:
                for i in range(B):
                    final_pred_text, target_text, final_pred_tokens, target_tokens = log_predict_targets(tokenizer, target[i], prediction[i])
                    samples[batch["sample_id"][i]] = {
                        **samples[batch["sample_id"][i]],
                        "final_pred_text": final_pred_text,
                        "final_pred_tokens": f"{final_pred_tokens}",
                        "target_text": target_text,
                        "target_tokens": f"{target_tokens}"
                    }

        # ✅ Evaluate once at end of epoch
        eval_avg_loss, eval_avg_accuracy = evaluate(model, tokenizer, eval_dataloader, device)

        avg_train_loss = running_loss / total_batches if total_batches > 0 else 0
        avg_train_accuracy = running_accuracy / total_batches if total_batches > 0 else 0

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_accuracy,
            "eval_loss": eval_avg_loss,
            "eval_accuracy": eval_avg_accuracy
        })

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_accuracy:.4f}")
        print(f"Eval  Loss: {eval_avg_loss:.4f} | Eval  Acc: {eval_avg_accuracy:.4f}")

    # ✅ Final training samples table
    text_table = wandb.Table(columns=["sample_id", "first_predicted_text", "first_predicted_tokens", "last_predicted", "target", "last_predicted_tokens", "target_tokens"])
    for sample_id, data in samples.items():
        text_table.add_data(
            sample_id,
            data.get("first_pred_text", ""),
            data.get("first_pred_tokens", ""),
            data.get("final_pred_text", ""),
            data.get("target_text", ""),
            data.get("final_pred_tokens", ""),
            data.get("target_tokens", "")
        )

    wandb.log({"training_text": text_table})


if __name__ == "__main__":
    init_wandb()
    device = get_device()
    model = whisper.load_model("tiny").to(device)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

   # train_data = load_dataset("EthanGLEdwards/welsh-transcription-samples")['train']
    train_data = load_dataset("ClemSummer/welsh-transcription-samples-7k")['train']
    
    train_dataset = AudioDataset(train_data, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=whisper_collate_fn,
        num_workers=4  # ✅ Use multiple workers!
    )

    #eval_data = load_dataset("EthanGLEdwards/welsh-transcription-samples")['validation']
    eval_data = load_dataset("ClemSummer/welsh-transcription-samples-7k")['validation']

    eval_dataset = AudioDataset(eval_data, tokenizer)
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=whisper_collate_fn,
        num_workers=2  # ✅ Same here
    )

    train(model, tokenizer, train_dataloader, eval_dataloader, device)
    torch.save(model.state_dict(), "fine_tuned_welsh_model.pth")

    wandb.finish()