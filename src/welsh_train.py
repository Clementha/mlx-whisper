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
    audio = example["audio"]["array"]  # NumPy array
    sr = example["audio"]["sampling_rate"]
    if sr != 16000:
        waveform = torch.tensor(audio).unsqueeze(0)  # (1, samples)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=exp_sr)
        audio = resampler(waveform).squeeze().numpy()

    # Pad or trim to 30s as expected by Whisper
    audio = whisper.pad_or_trim(audio)
    return whisper.log_mel_spectrogram(audio)

def evaluate(model, tokenizer, eval_dataloader, batch_idx):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0
    
    # Collect predictions and targets for wandb table
    val_samples = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            audio_batch = batch["mel"]
            eval_tokens = batch["caption_ids"]
            sample_ids = batch["sample_id"] if "sample_id" in batch else [str(i) for i in range(audio_batch.size(0))]

            B = audio_batch.size(0)
            target = eval_tokens[:, 1:].contiguous()  # [B, T-1]

            prediction = model(tokens=eval_tokens, mel=audio_batch)  # [B, T, Vocab]

            loss = criterion(prediction[:, :-1, :].transpose(1, 2), target)
            total_loss += loss.item()

            batch_accuracy = compute_avg_masked_accuracy_per_batch(prediction, target, B)
            total_accuracy += batch_accuracy

            total_batches += 1

            # Collect predictions and targets for wandb table
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
    print(f"Evaluation - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

    # Log validation_text table to wandb
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


def train(model, tokenizer, train_dataloader, eval_dataloader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    samples = {}
    for epoch in range(EPOCHS):
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            mels = batch["mel"]#.to(device)  # [B, N]
            caption_ids = batch["caption_ids"]#.to(device)
            B = mels.size(0)
            target = caption_ids[:, 1:].contiguous()  # [B, T-1]
            
            
            prediction = model(tokens=caption_ids, mel=mels)  # [B, T, Vocab]
            avg_batch_accuracy = compute_avg_masked_accuracy_per_batch(prediction, target, B)
            if epoch == 0:
                for i in range(B):

                    first_pred_text, _, first_pred_tokens, _ = log_predict_targets(tokenizer, target[i], prediction[i])

                    samples[batch["sample_id"][i]] = { "first_pred_text": first_pred_text,
                                                        "first_pred_tokens": f"{first_pred_tokens}" }
    

            loss = criterion(prediction[:, :-1, :].transpose(1, 2), target)    # [B, V, T] vs [B, T]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            if epoch == EPOCHS - 1:
                for i in range(B):
                    final_pred_text, target_text, final_pred_tokens, target_tokens = log_predict_targets(tokenizer, target[i], prediction[i])
                    samples[batch["sample_id"][i]] = {**samples[batch["sample_id"][i]], "final_pred_text": final_pred_text,
                                                        "final_pred_tokens": f"{final_pred_tokens}",
                                                         "target_text": target_text,
                                                         "target_tokens": f"{target_tokens}" }
            eval_avg_loss, eval_avg_accuracy = evaluate(model, tokenizer, eval_dataloader, batch_idx)
            # wandb.log({"epoch": epoch + 1, "loss": loss.item(), "training_text": text_table, "avg_batch_accuracy": avg_batch_accuracy, "avg_whisper_accuracy": avg_whisper_accuracy, "eval_avg_loss": eval_avg_loss, "eval_avg_accuracy": eval_avg_accuracy, "eval_text_table": eval_text_table })
     
            wandb.log({
                "loss": loss.item(),
                "avg_batch_accuracy": avg_batch_accuracy,
                "eval_avg_loss": eval_avg_loss,
                "eval_avg_accuracy": eval_avg_accuracy
            })
    text_table = wandb.Table(columns=["sample_id", "first_predicted_text", "first_predicted_tokens", "last_predicted", "target", "last_predicted_tokens", "target_tokens"])
    for sample_id, data in samples.items():
        ## TODO: Could remove non text tokens from the tokens and predictions.
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
    # device = get_device()
    model = whisper.load_model("tiny")#, device=device)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

    train_data = load_dataset("EthanGLEdwards/welsh-transcription-samples")['train']
    train_dataset = AudioDataset(train_data, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=whisper_collate_fn
    )

    eval_data = load_dataset("EthanGLEdwards/welsh-transcription-samples")['validation']

    eval_dataset = AudioDataset(eval_data, tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=whisper_collate_fn)

    train(model, tokenizer, train_dataloader, eval_dataloader)
    torch.save(model.state_dict(), "fine_tuned_welsh_model.pth")

    wandb.finish()