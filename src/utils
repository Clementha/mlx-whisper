import torch
import wandb
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(override=True)

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    return device

def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

def init_wandb(config={}):
    default_config = {
        "learning_rate": 0.001,
        "architecture": "CNN-1D",
        "dataset": "urban-sound-dataset",
        "epochs": 5,
    }
    # Start a new wandb run to track this script.
    return wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=os.environ.get("WANDB_ENTITY", "audio-party"),
        # Set the wandb project where this run will be logged.
        project=os.environ.get("WANDB_PROJECT", "mlx-whisper"),
        # Track hyperparameters and run metadata.
        config={**default_config, **config},
    )

def save_artifact(artifact_name, artifact_description, file_extension='pt', type="model"):
    artifact = wandb.Artifact(
        name=artifact_name,
        type=type,
        description=artifact_description
    )
    artifact.add_file(f"./data/{artifact_name}.{file_extension}")
    wandb.log_artifact(artifact)

def load_artifact_path(artifact_name, version="latest", file_extension='pt'):
    artifact = wandb.use_artifact(f"{artifact_name}:{version}")
    directory = artifact.download()
    return f"{directory}/{artifact_name}.{file_extension}"

def load_model_path(model_name):
    downloaded_model_path = wandb.use_model(model_name)
    return downloaded_model_path