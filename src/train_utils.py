import whisper
import torch

def whisper_without_fine_tuning(model, audio_batch, device):
    options = whisper.DecodingOptions()
    outputs = []
    for audio in audio_batch:
        log_mel = whisper.log_mel_spectrogram(audio).to(device)
        response = whisper.decode(model, log_mel, options)
        outputs.append(response.text)
    return outputs

def log_without_fine_tuning(model, audio_batch, wandb_pre_fine_tune_logs):
    print("Whisper predictions (no fine-tuning):")
    predictions_raw = whisper_without_fine_tuning(model, audio_batch, model.device)
    for i, text in enumerate(predictions_raw):
        wandb_pre_fine_tune_logs.append(text)
        print(f"  Sample {i + 1}: {text}")

def log_predict_targets(text_table, tokenizer, wandb_pre_fine_tune_logs, target, prediction, batch_num):
    for i in range(batch_num):
        print(f"Sample {i + 1}:")
        pred_tokens = torch.argmax(prediction[:, :-1, :], dim=-1).contiguous()  # [B, T-1]
        target_text = tokenizer.decode(target[i].tolist())
        predicted_text = tokenizer.decode(pred_tokens[i].tolist())
        target_tokens = f"{target[i].squeeze().tolist()}"
        prediction_tokens = f"{torch.argmax(prediction[i], dim=-1).squeeze().tolist()}"
        text_table.add_data(
            f"Sample {i + 1}:",
            wandb_pre_fine_tune_logs[i],
            predicted_text,
            target_text,
            prediction_tokens,
            target_tokens
        )
        print("  Target text:   ", target_text)
        print("  Predicted text:", predicted_text)