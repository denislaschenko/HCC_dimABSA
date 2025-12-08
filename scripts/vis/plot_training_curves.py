import os
import matplotlib.pyplot as plt


def plot_training_curves(train_losses, val_losses, output_dir):
    """
    Plot training and validation loss across epochs.

    Args:
        train_losses (list of float): Training loss per epoch.
        val_losses (list of float): Validation loss per epoch.
        output_dir (str): Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, marker='o', label='Training Loss')
    plt.plot(epochs, val_losses, marker='o', label='Validation Loss')

    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path)
    plt.close()

    print(f"📉 Loss curve saved to: {save_path}")
