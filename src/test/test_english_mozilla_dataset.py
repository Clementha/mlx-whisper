from datasets import load_dataset

# Load Welsh subset, validated split (use 'train' or 'test' too)
ds = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "en",   # Welsh language code
    split="train",
      #use_auth_token=True  # 'train', 'validation', 'test'
)

print(ds)
print(ds[0])

