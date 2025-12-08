import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.subtask_1 import config

df = pd.read_json(os.path.join(config.PREDICTION_SUBDIR, "train_set_residual_analysis.jsonl"), lines=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(data=df, x="Valence", y="Resid_Valence", ax=axes[0], alpha=0.5, color='purple')
axes[0].axhline(0, color='r', linestyle='--', label="Zero Error (Perfect)")
axes[0].set_title("Valence Analysis")
axes[0].set_xlabel("Gold Label")
axes[0].set_ylabel("Residuals")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

sns.scatterplot(data=df, x="Arousal", y="Resid_Arousal", ax=axes[1], alpha=0.5, color='orange')
axes[1].axhline(0, color='r', linestyle='--', label="Zero Error (Perfect)")
axes[1].set_title("Arousal Analysis")
axes[1].set_xlabel("Gold Label")
axes[1].set_ylabel("Residuals")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()