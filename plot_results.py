import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/results.csv")

plt.figure(figsize=(10, 6))
plt.bar(
    df["configuration"],
    df["best_validation_accuracy"]
)
plt.xlabel("Augmentation Configuration")
plt.ylabel("Validation Accuracy")
plt.title("MNIST Data Augmentation Ablation Study")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("results/comparison.png", dpi=200)
plt.show()



configs = [
    "baseline",
    "rotation",
    "rotation_shift",
    "rotation_shift_noise"
]

for config in configs:
    df = pd.read_csv(f"results/history/{config}.csv")
    
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["epoch"],
        df["val_accuracy"],
        label=config
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"{config} Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{config}_curve.png")
    plt.show()