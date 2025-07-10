import wandb
from utils import init_wandb, save_artifact

# Create a data directory and move the .pth file into that data directory

MODEL_FILE_NAME = "fine_tuned_welsh_model"
init_wandb()

save_artifact(MODEL_FILE_NAME, "Model trained on 20 epochs", "pth")

wandb.finish()