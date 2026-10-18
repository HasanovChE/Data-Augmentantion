import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import get_datasets
from augmentations import (
    rotation_only,
    rotation_shift,
    rotation_shift_noise
)
from train import train_model

EPOCHS = 5
BATCH_SIZE = 64
SEED = 42

CONFIGS = {
    "baseline": None,
    "rotation": rotation_only(),
    "rotation_shift": rotation_shift(),
    "rotation_shift_noise": rotation_shift_noise()
}

if __name__ == '__main__':
    os.makedirs("results/models", exist_ok=True)
    csv_path = "results/results.csv"

    all_results = []

    for name, augmentation in CONFIGS.items():
        print()
        print("=" * 60)
        print(f"RUNNING: {name}")
        print("=" * 60)
        
        train_dataset, test_dataset = get_datasets(augmentation)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
        
        model, history, training_time = train_model(
            train_loader,
            test_loader,
            epochs=EPOCHS,
            seed=SEED
        )
        
        final_accuracy = history[-1]["val_accuracy"]
        best_accuracy = max(h["val_accuracy"] for h in history)
        
        model_path = f"results/models/{name}.pt"
        torch.save(model.state_dict(), model_path)
        
        row = {
            "configuration": name,
            "final_validation_accuracy": final_accuracy,
            "best_validation_accuracy": best_accuracy,
            "training_time_seconds": training_time,
            "epochs": EPOCHS,
            "seed": SEED
        }
        all_results.append(row)

        pd.DataFrame(all_results).to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("ALL 4 MODELS TRAINED SUCCESSFULLY FROM SCRATCH!")
    print("=" * 60)
    print(pd.read_csv(csv_path))