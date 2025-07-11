from datasets import load_dataset, Dataset, DatasetDict, Audio
from huggingface_hub import HfApi
import os

# Take first 500 with islice
from itertools import islice

# ---------------------------
# CONFIG
# ---------------------------
LANGUAGE = "en"
SPEAKER = "Mozilla"
REPO_ID = "ClemSummer/english-transcription-samples"  # change to your username/repo name!

# ---------------------------
# 1. Load original Common Voice Welsh split
# ---------------------------
cv = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    LANGUAGE,
    split="train",
    streaming=True,
    trust_remote_code=True
)

cv_small = list(islice(cv, 500))
cv_small = Dataset.from_list(cv_small)
print(f"Loaded {len(cv_small)} rows.")

# ---------------------------
# 2. Map to new columns
# ---------------------------
def map_fn(example, idx):
    return {
        "audio": example["audio"],          # keep original audio
        "caption": example["sentence"],     # rename 'sentence' to 'caption'
        "language": LANGUAGE,
        "speaker": SPEAKER,
        "sample_id": idx                    # unique index
    }

# Use .map() with batched=False and with_indices=True


cv_mapped = cv_small.map(
    map_fn,
    with_indices=True,
    remove_columns=cv.column_names
)
cv_mapped = cv_mapped.select(range(500)) # Limit to 500 rows for testing
print(cv_mapped)
print(cv_mapped[0])

# ---------------------------
# 3. Cast audio column (optional but recommended)
# ---------------------------
cv_mapped = cv_mapped.cast_column("audio", Audio())

# ---------------------------
# 4. Save to disk for backup (optional)
# ---------------------------
cv_mapped.save_to_disk("./my_english_dataset")
print("Saved locally to ./my_english_dataset")

# ---------------------------
# 5. Push to your own Hugging Face repo
# ---------------------------
cv_mapped.push_to_hub(REPO_ID)
print(f"Pushed to https://huggingface.co/datasets/{REPO_ID}")

# ✅ Done!