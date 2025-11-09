import pandas as pd
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "results.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "performance_plot.png")

def generate_plot():
    try:
        data = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find results file at {CSV_FILE}")
        return

    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')

    x_labels = data['date'].dt.strftime('%Y-%m-%d') + '\n(' + data['experiment'] + ')'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Subtask 1: Model Performance Over Time', fontsize=18, y=0.95)

    ax1.plot(x_labels, data['pcc_v'], 'o-', label='PCC Valence (↑)', color='#377eb8', linewidth=2, markersize=8)
    ax1.plot(x_labels, data['pcc_a'], 's-', label='PCC Arousal (↑)', color='#ff7f00', linewidth=2, markersize=8)
    ax1.set_ylabel('PCC Score (Higher is Better)', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_title('Correlation (PCC)', fontsize=14)

    ax2.plot(x_labels, data['rmse_va'], '^-', label='RMSE VA (↓)', color='#e41a1c', linewidth=2, markersize=8)
    ax2.set_ylabel('RMSE Score (Lower is Better)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title('Error (RMSE)', fontsize=14)

    plt.xlabel('Experiment (Date & Version)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    try:
        plt.savefig(OUTPUT_FILE)
        print(f"Successfully generated and saved plot from CSV to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
        generate_plot()