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
        # audio_array = self.dataset[idx]["audio"]["array"]
        # audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        # Pad or trim to 30s (Whisper expects 30s input at 16kHz)
        # audio_tensor = whisper.pad_or_trim(audio_tensor)
        mel = preprocess_audio(self.dataset[idx])
        # print("train_tokens_batch.shape: ", caption_ids.shape)

        # return (audio_tensor, caption_ids)
        return {
        "mel": mel,
        "caption_ids": caption_ids,
        "sample_id": id
    }

def whisper_collate_fn(batch):
    mels = [item["mel"] for item in batch]  # Already log-mel spectrogram
    caption_ids = [item["caption_ids"] for item in batch]  # Already tokenized
    sample_ids = [item["sample_id"] for item in batch]  # Keep sample IDs
    # Stack mel spectrograms (B, 80, 3000)
    mels = torch.stack(mels)

    # Pad caption_ids to the max length in the batch (B, max_len)
    caption_ids = pad_sequence(caption_ids, batch_first=True, padding_value=caption_ids[0][-1].item())  # Use EOT as pad

    return {
        "mel": mels,
        "caption_ids": caption_ids,
        "sample_id": sample_ids  # Keep sample IDs
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
    samples = {}
    for epoch in range(EPOCHS):
        # print(f"\n---- Epoch {epoch + 1}/{EPOCHS} ----")
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            mels = batch["mel"]#.to(device)  # [B, N]
            caption_ids = batch["caption_ids"]#.to(device)

            # mel = whisper.log_mel_spectrogram(audio_batch)#.to(device)  # [B, 80, T]

            B = mels.size(0)
            target = caption_ids[:, 1:].contiguous()  # [B, T-1]
            
            prediction = model(tokens=caption_ids, mel=mels)  # [B, T, Vocab]
            if epoch == 0:
                for i in range(B):

                    first_pred_text, _, first_pred_tokens, _ = log_predict_targets(tokenizer, target[i], prediction[i])

                    samples[batch["sample_id"][i]] = { "first_pred_text": first_pred_text,
                                                        "first_pred_tokens": f"{first_pred_tokens}" }
    

            loss = criterion(prediction[:, :-1, :].transpose(1, 2), target)    # [B, V, T] vs [B, T]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_batch_accuracy = compute_avg_masked_accuracy_per_batch(prediction, target, B)
            
            if epoch == EPOCHS - 1:
                for i in range(B):
                    final_pred_text, target_text, final_pred_tokens, target_tokens = log_predict_targets(tokenizer, target[i], prediction[i])
                    samples[batch["sample_id"][i]] = {**samples[batch["sample_id"][i]], "final_pred_text": final_pred_text,
                                                        "final_pred_tokens": f"{final_pred_tokens}",
                                                         "target_text": target_text,
                                                         "target_tokens": f"{target_tokens}" }
                    
            wandb.log({
                "epoch": epoch + 1,
                "batch_idx": batch_idx + 1,
                "loss": loss.item(),
                "avg_batch_accuracy": avg_batch_accuracy
            })
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
    
    # text_table.add_data(
    #     "First Prediction",
    #     first_pred_text,
    #     "Target",
    #     "First Predicted Tokens",
    #     "Target Tokens"
    # )

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
    batch_size=4,
    shuffle=True,
    collate_fn=whisper_collate_fn
)
    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # eval_dataset = AudioDataset(["audio/Ethan--Bes.m4a"])
    # eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train(model, tokenizer, train_dataloader)
    torch.save(model.state_dict(), "fine_tuned_welsh_model.pth")

    wandb.finish()