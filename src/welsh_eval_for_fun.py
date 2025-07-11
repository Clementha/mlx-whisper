import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import whisper
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from welsh_train_utils import (
    get_device, compute_avg_masked_accuracy_per_batch,
    log_predict_targets, gen_token_ids_with_special_tokens
)

# ------------------------------------------
# YOUR CONFIG
# ------------------------------------------
DEFAULT_EPOCHS = 5
DEFAULT_LR = 1e-5
BATCH_SIZE = 3
# DEFAULT_DATASET_SIZE = 6240  # ✅ Max train rows available

# ------------------------------------------
# EVALUATE
# ------------------------------------------
def evaluate(model, tokenizer, eval_dataloader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            mels = batch["mel"].to(device)
            caption_ids = batch["caption_ids"].to(device)
            target = caption_ids[:, 1:]

            prediction = model(tokens=caption_ids, mel=mels)
            loss = criterion(prediction[:, :-1, :].transpose(1, 2), target)
            acc = compute_avg_masked_accuracy_per_batch(prediction, target, mels.size(0))

            total_loss += loss.item()
            total_acc += acc
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_acc = total_acc / total_batches if total_batches > 0 else 0

    return avg_loss, avg_acc

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


# ------------------------------------------
# MAIN ENTRYPOINT
# ------------------------------------------
def main():
    wandb.init(
        project="whisper-welsh-finetune",
        config={
            "epochs": DEFAULT_EPOCHS,
            "learning_rate": DEFAULT_LR
        }
    )
    config = wandb.config
    #print(f"Config: {config}")
    device = get_device()
    print(f"Using device: {device}")

    model = whisper.load_model("tiny").to(device)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

    # ✅ Load your dataset & splits
    dataset_name = "ClemSummer/welsh-transcription-samples-7k"
    ds = load_dataset(dataset_name)

    en_dataset_name = "ClemSummer/english-transcription-samples"
    en_ds = load_dataset(en_dataset_name)

    # ✅ Limit train data based on sweep param
    # dataset_size = min(config.dataset_size, DEFAULT_DATASET_SIZE)
    # dataset_size = DEFAULT_DATASET_SIZE
    # train_data = ds["train"].select(range(dataset_size))
    train_data = ds["train"]
    eval_data = ds["validation"]  # Keep full eval split (10%)
    train_dataset = AudioDataset(train_data, tokenizer)
    eval_dataset = AudioDataset(eval_data, tokenizer)

    en_train_data = en_ds["train"]
    en_eval_data = en_ds["validation"]
    en_train_dataset = AudioDataset(en_train_data, tokenizer)
    en_eval_dataset = AudioDataset(en_eval_data, tokenizer) 

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=whisper_collate_fn,
        num_workers=4
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=whisper_collate_fn,
        num_workers=2
    )

    en_eval_dataloader = DataLoader(
        en_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=whisper_collate_fn,
        num_workers=2
    )

    # train(
    #     model,
    #     tokenizer,
    #     train_dataloader,
    #     eval_dataloader,
    #     device,
    #     config.epochs,
    #     config.learning_rate
    # )
    # model_filename = f"fine_tuned_welsh_epochs_{config.epochs}.pth"
    # torch.save(model.state_dict(), model_filename)
    # print(f"Saved model to {model_filename}")
    # wandb.finish()
    print("Starting evaluation of original tiny...")
    eval_loss, eval_acc = evaluate(model, tokenizer, eval_dataloader, device)
    print(f"Original eval_loss: {eval_loss:.4f}, eval_acc: {eval_acc:.4f}")
    eval_loss, eval_acc = evaluate(model, tokenizer, en_eval_dataloader, device)
    print(f"Original en_eval_loss: {eval_loss:.4f}, en_eval_acc: {eval_acc:.4f}")

    print("Evaluation of tuned model")
    model.load_state_dict(torch.load("./fine_tuned_welsh_model.pth"))
    eval_loss, eval_acc = evaluate(model, tokenizer, eval_dataloader, device)
    print(f"Tuned eval_loss: {eval_loss:.4f}, eval_acc: {eval_acc:.4f}")
    eval_loss, eval_acc = evaluate(model, tokenizer, en_eval_dataloader, device)
    print(f"Original en_eval_loss: {eval_loss:.4f}, en_eval_acc: {eval_acc:.4f}")   

# ------------------------------------------
# SCRIPT ENTRY
# ------------------------------------------
if __name__ == "__main__":
    main()