import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.shared import config

df = pd.read_json(os.path.join(config.PREDICTION_SUBDIR, "train_set_residual_analysis.jsonl"), lines=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(data=df, x="Valence", y="Predicted_Valence", ax=axes[0], alpha=0.5, color='blue')
axes[0].plot([0, 10], [0, 10], 'r--', label="Perfect Prediction")
axes[0].set_title("Valence Analysis")
axes[0].set_xlabel("Gold Label")
axes[0].set_ylabel("Predicted Label")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

sns.scatterplot(data=df, x="Arousal", y="Predicted_Arousal", ax=axes[1], alpha=0.5, color='green')
axes[1].plot([0, 10], [0, 10], 'r--', label="Perfect Prediction")
axes[1].set_title("Arousal Analysis")
axes[1].set_xlabel("Gold Label")
axes[1].set_ylabel("Predicted Label")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()