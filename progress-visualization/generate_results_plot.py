import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("results.csv")

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True)
fig.suptitle('Model performance over time', fontsize=16)

ax1.plot(data['date'], data['pcc_v'], 'o-', label='PCC Valence (↑)')
ax1.plot(data['date'], data['pcc_a'], 's-', label='PCC Arousal (↑)')
ax1.set_ylabel('PCC Score')
ax1.set_ylim(0, 1)
ax1.legend()
ax1.grid(True, linestyle='--')

ax2.plot(data['date'], data['rmse_va'], '^-', label='RMSE VA (↓)', color='red')
ax2.set_ylabel('RMSE Score')
ax2.legend()
ax2.grid(True, linestyle='--')

plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig("performance_plot.png")
print("performance_plot.png has been updated.")
