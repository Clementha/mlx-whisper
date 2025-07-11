import whisper
import torch
import torch.nn.functional as F

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    return device

def log_predict_targets(tokenizer, target, prediction):
    pred_tokens = torch.argmax(prediction[:-1, :], dim=-1).contiguous()  # [B, T-1]
    target_text = tokenizer.decode(target.tolist())
    predicted_text = tokenizer.decode(pred_tokens.tolist())
    target_tokens = target.squeeze().tolist()
    prediction_tokens = torch.argmax(prediction, dim=-1).squeeze().tolist()
    return predicted_text, target_text, prediction_tokens, target_tokens


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
    return average_accuracy


def gen_token_ids_with_special_tokens(tokenizer, language, text):
    ids = []
    ids += [tokenizer.sot]
    if language == "cy":
        ids += [50297]
    elif language == "en":
        ids += [tokenizer.language_token]
    else:
        raise ValueError(f"Unsupported language: {language}")
    ids += [tokenizer.transcribe]
    ids += [tokenizer.no_timestamps]
    ids += tokenizer.encode(f" {text}")
    ids += [tokenizer.eot]
    return torch.tensor(ids, dtype=torch.long)