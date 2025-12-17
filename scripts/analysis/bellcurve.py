import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.getcwd())

from src.shared import config


def plot_distributions():
    input_file = os.path.join(config.PREDICTION_SUBDIR, "train_set_residual_analysis.jsonl")

    if not os.path.exists(input_file):
        print(f"Error: File not found at {input_file}")
        print("Please run the 'run_model.py' script first to generate the data.")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_json(input_file, lines=True, orient='records')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_style("whitegrid")

    sns.histplot(data=df, x="Valence", kde=True, ax=axes[0], color='blue', bins=20)
    axes[0].set_title("Distribution of Gold Valence Labels")
    axes[0].set_xlabel("Valence Score")
    axes[0].set_ylabel("Count")

    v_mean = df["Valence"].mean()
    v_std = df["Valence"].std()
    axes[0].axvline(v_mean, color='red', linestyle='--', label=f"Mean: {v_mean:.2f}")
    axes[0].legend()

    sns.histplot(data=df, x="Arousal", kde=True, ax=axes[1], color='green', bins=20)
    axes[1].set_title("Distribution of Gold Arousal Labels")
    axes[1].set_xlabel("Arousal Score")
    axes[1].set_ylabel("Count")

    a_mean = df["Arousal"].mean()
    a_std = df["Arousal"].std()
    axes[1].axvline(a_mean, color='red', linestyle='--', label=f"Mean: {a_mean:.2f}")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print("\n--- Data Statistics ---")
    print(df[["Valence", "Arousal"]].describe())

    print("\n--- Skewness (0 = Normal, Positive = Tail on right, Negative = Tail on left) ---")
    print(df[["Valence", "Arousal"]].skew())


if __name__ == "__main__":
    plot_distributions()