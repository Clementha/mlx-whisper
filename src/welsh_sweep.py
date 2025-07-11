import wandb

sweep_config = {
    'method': 'grid',
    'program': 'welsh_train.py',
    'parameters': {
        'epochs': {'values': [0, 5, 10]},
        'dataset_size': {'values': [500, 3000, 6240]}
    }
}

print("Creating sweep now...")
sweep_id = wandb.sweep(sweep_config, project="whisper-welsh-sweep")
print(f"Sweep ID: {sweep_id}")
