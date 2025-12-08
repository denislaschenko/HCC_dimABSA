import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from src.subtask_1 import config

df = pd.read_json(os.path.join(config.PREDICTION_SUBDIR, "train_set_residual_analysis.jsonl"), lines=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

stats.probplot(df['Resid_Valence'], dist="norm", plot=axes[0])
axes[0].set_title("Valence Residuals Q-Q Plot")
axes[0].grid(True, alpha=0.3)

stats.probplot(df['Resid_Arousal'], dist="norm", plot=axes[1])
axes[1].set_title("Arousal Residuals Q-Q Plot")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()