from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# csv with the loss, accuracy and duration data (per checkpointed epoch)
LOSS_FILENAME = "loss.csv"

# Figure will be saved at OUTPUT_PLOT_DIR / OUTPUT_PLOT_FILENAME
OUTPUT_PLOT_DIR = Path("plots")
OUTPUT_PLOT_FILENAME = "training_metrics.png"

if not OUTPUT_PLOT_DIR.exists():
    OUTPUT_PLOT_DIR.mkdir()
save_path = OUTPUT_PLOT_DIR / OUTPUT_PLOT_FILENAME

# Load the data
df = pd.read_csv(LOSS_FILENAME)

# Set the visual style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

# Create a subplot figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss
sns.lineplot(data=df, x="epoch", y="loss", marker="o", color="royalblue", ax=ax1)
ax1.set_title("Training Loss per Epoch", fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")

# Plot Accuracy
sns.lineplot(data=df, x="epoch", y="accuracy", marker="s", color="darkorange", ax=ax2)
ax2.set_title("Training Accuracy per Epoch", fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1)  # Accuracy usually ranges from 0 to 1

# Adjust layout and show
plt.tight_layout()

# Save the figure
plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"Plot saved successfully to: {save_path}")

plt.show()
