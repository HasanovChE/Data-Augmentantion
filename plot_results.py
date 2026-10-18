import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "results/results.csv"

if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found! Please run experiment.py first.")
else:
    df = pd.read_csv(csv_path)

    os.makedirs("results", exist_ok=True)

    # -------------------------------------------------------------
    # CHART 1: Single Bar Chart for Best Validation Accuracy
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        df["configuration"],
        df["best_validation_accuracy"],
        color=['#2b5c8f', '#4682b4', '#6897bb', '#87ceeb']
    )
    plt.xlabel("Augmentation Configuration")
    plt.ylabel("Validation Accuracy")
    plt.title("MNIST Data Augmentation Ablation Study - Accuracy")
    
    # Dynamic Y-axis limits to clearly highlight small differences
    min_acc = df["best_validation_accuracy"].min()
    plt.ylim(min_acc - 0.003, 1.0)
    plt.xticks(rotation=20)

    # Display exact accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.0003,
            f"{height:.4f}",
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=200)
    print("Saved accuracy comparison chart to: results/comparison.png")
    plt.close()

    # -------------------------------------------------------------
    # CHART 2: Subplot Comparing Accuracy and Training Time
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Validation Accuracy
    bars1 = ax1.bar(df["configuration"], df["best_validation_accuracy"], color='#4682b4')
    ax1.set_title("Best Validation Accuracy")
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(min_acc - 0.003, 1.0)
    ax1.tick_params(axis='x', rotation=20)
    for p in bars1:
        h = p.get_height()
        ax1.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h),
                     ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')

    # Subplot 2: Training Time (Seconds)
    bars2 = ax2.bar(df["configuration"], df["training_time_seconds"], color='#e74c3c')
    ax2.set_title("Training Time (Seconds)")
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Time (s)")
    ax2.tick_params(axis='x', rotation=20)
    for p in bars2:
        h = p.get_height()
        ax2.annotate(f"{h:.1f}s", (p.get_x() + p.get_width() / 2., h),
                     ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')

    plt.suptitle("MNIST Ablation Study - Accuracy vs Training Time", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/metrics_comparison.png", dpi=200)
    print("Saved metrics summary chart to: results/metrics_comparison.png")
    plt.close()

    print("All charts generated successfully!")