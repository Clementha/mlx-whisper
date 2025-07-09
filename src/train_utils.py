import whisper
import torch
from torch.nn.utils.rnn import pad_sequence

def _whisper_without_fine_tuning(model, audio_batch, device):
    options = whisper.DecodingOptions()
    outputs = []
    for audio in audio_batch:
        log_mel = whisper.log_mel_spectrogram(audio).to(device)
        response = whisper.decode(model, log_mel, options)
        outputs.append(response.text)
    return outputs

def log_without_fine_tuning(model, audio_batch, wandb_pre_fine_tune_logs):
    print("Whisper predictions (no fine-tuning):")
    predictions_raw = _whisper_without_fine_tuning(model, audio_batch, model.device)
    for i, text in enumerate(predictions_raw):
        wandb_pre_fine_tune_logs.append(text)
        print(f"  Sample {i + 1}: {text}")

def log_predict_targets(text_table, tokenizer, wandb_pre_fine_tune_logs, target, prediction, batch_num):
    for i in range(batch_num):
        print(f"Sample {i + 1}:")
        pred_tokens = torch.argmax(prediction[:, :-1, :], dim=-1).contiguous()  # [B, T-1]
        target_text = tokenizer.decode(target[i].tolist())
        predicted_text = tokenizer.decode(pred_tokens[i].tolist())
        target_tokens = target[i].squeeze().tolist()
        prediction_tokens = torch.argmax(prediction[i], dim=-1).squeeze().tolist()
        text_table.add_data(
            f"Sample {i + 1}:",
            wandb_pre_fine_tune_logs[i],
            predicted_text,
            target_text,
            f"{prediction_tokens}",
            f"{target_tokens}"
        )
        print("  Target text:   ", target_text)
        print("  Predicted text:", predicted_text)


PAD_TOKEN = 0

def compute_avg_masked_accuracy_per_batch(prediction, target, batch_num):
    all_batch_accuracies = []
    for i in range(batch_num):
        pred_tokens = torch.argmax(prediction[i], dim=-1).squeeze()
        target_tokens = target[i].squeeze()
        # Handle padding and mask
        max_len = max(pred_tokens.shape[0], target_tokens.shape[0])
        pred_padded = torch.full((max_len,), PAD_TOKEN, dtype=pred_tokens.dtype, device=pred_tokens.device)
        target_padded = torch.full((max_len,), PAD_TOKEN, dtype=target_tokens.dtype, device=target_tokens.device)
        pred_padded[:pred_tokens.shape[0]] = pred_tokens
        target_padded[:target_tokens.shape[0]] = target_tokens
        mask = target_padded != PAD_TOKEN
        correct = (pred_padded == target_padded) & mask

        batch_correct = correct.sum().item()
        batch_total = mask.sum().item()

        batch_accuracy = (batch_correct / batch_total) * 100 if batch_total > 0 else 0.0
        all_batch_accuracies.append(batch_accuracy)

    average_accuracy = sum(all_batch_accuracies) / len(all_batch_accuracies)
    print(f"Average accuracy of the batch: {average_accuracy:.2f}%")
    return average_accuracy