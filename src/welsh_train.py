from datasets import load_dataset
import torch
import whisper
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import get_device, init_wandb
from template_train_utils import log_without_fine_tuning, log_predict_targets, compute_avg_masked_accuracy_per_batch, average_whisper_accuracy_before_ft, gen_token_ids_with_special_tokens
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
        caption_ids = gen_token_ids_with_special_tokens(self.tokenizer, caption)
        # audio_array = self.dataset[idx]["audio"]["array"]
        # audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        # Pad or trim to 30s (Whisper expects 30s input at 16kHz)
        # audio_tensor = whisper.pad_or_trim(audio_tensor)
        mel = preprocess_audio(self.dataset[idx])
        print("train_tokens_batch.shape: ", caption_ids.shape)

        # return (audio_tensor, caption_ids)
        return {
        "mel": mel,
        "caption_ids": caption_ids
    }

def whisper_collate_fn(batch):
    mels = [item["mel"] for item in batch]  # Already log-mel spectrogram
    caption_ids = [item["caption_ids"] for item in batch]  # Already tokenized

    # Stack mel spectrograms (B, 80, 3000)
    mels = torch.stack(mels)

    # Pad caption_ids to the max length in the batch (B, max_len)
    caption_ids = pad_sequence(caption_ids, batch_first=True, padding_value=caption_ids[0][-1].item())  # Use EOT as pad

    return {
        "mel": mels,
        "caption_ids": caption_ids
    }

# Whisper expects 16kHz mono audio
expected_sr = 16000

# Extract the raw audio and resample if necessary
def preprocess_audio(example):
    audio = example["audio"]["array"]  # NumPy array
    sr = example["audio"]["sampling_rate"]

    # Resample if not 16kHz
    if sr != expected_sr:

        waveform = torch.tensor(audio).unsqueeze(0)  # (1, samples)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=expected_sr)
        audio = resampler(waveform).squeeze().numpy()

    # Pad or trim to 30s as expected by Whisper
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio)

    return mel

def train(model, tokenizer, train_dataloader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    wandb_pre_fine_tune_logs = []
    for epoch in range(EPOCHS):
        text_table = wandb.Table(columns=["sample_num", "pre_fine_tuning", "last_predicted", "target", "last_predicted_tokens", "target_tokens"])
        print(f"\n---- Epoch {epoch + 1}/{EPOCHS} ----")
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            mels = batch["mel"]#.to(device)  # [B, N]
            caption_ids = batch["caption_ids"]#.to(device)

            # mel = whisper.log_mel_spectrogram(audio_batch)#.to(device)  # [B, 80, T]

            B = mels.size(0)
            target = caption_ids[:, 1:].contiguous()  # [B, T-1]
            
            # if epoch == 0 and batch_idx == 0:
            #     log_without_fine_tuning(model, mels, wandb_pre_fine_tune_logs)
            #     avg_whisper_accuracy = average_whisper_accuracy_before_ft(model, mels, target, tokenizer)
            
            prediction = model(tokens=caption_ids, mel=mels)  # [B, T, Vocab]
            loss = criterion(prediction[:, :-1, :].transpose(1, 2), target)    # [B, V, T] vs [B, T]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # avg_batch_accuracy = compute_avg_masked_accuracy_per_batch(prediction, target, B)
            
            # if epoch == EPOCHS - 1 and batch_idx == 0:
            #     log_predict_targets(text_table, tokenizer, wandb_pre_fine_tune_logs, target, prediction, B)
            
            # wandb.log({"epoch": epoch + 1, "loss": loss.item(), "training_text": text_table, "avg_batch_accuracy": avg_batch_accuracy, "avg_whisper_accuracy": avg_whisper_accuracy })


if __name__ == "__main__":
    # init_wandb()
    # device = get_device()
    model = whisper.load_model("tiny")#, device=device)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

    train_data = load_dataset("EthanGLEdwards/welsh-transcription-samples")['train']
    train_dataset = AudioDataset(train_data, tokenizer)
    train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=whisper_collate_fn
)
    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # eval_dataset = AudioDataset(["audio/Ethan--Bes.m4a"])
    # eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train(model, tokenizer, train_dataloader)
    torch.save(model.state_dict(), "fine_tuned_welsh_model.pth")

    # wandb.finish()