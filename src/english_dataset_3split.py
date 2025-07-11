from datasets import load_dataset, DatasetDict
REPO_ID_OLD = "ClemSummer/english-transcription-samplesOld"
REPO_ID = "ClemSummer/english-transcription-samples"
ds = load_dataset(REPO_ID_OLD, split="train")

split = ds.train_test_split(test_size=0.2, seed=42)
val_test = split['test'].train_test_split(test_size=0.5, seed=42)

ds_multi = DatasetDict({
    "train": split['train'],
    "validation": val_test['train'],
    "test": val_test['test']
})

ds_multi.push_to_hub(REPO_ID)