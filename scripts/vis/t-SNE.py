import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import Counter
from transformers import AutoTokenizer, AutoModel

# --- PATH SETUP ---
# Determine project root (../../ from scripts/analysis)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# --- CONFIGURATION ---
# Adjust these filenames if yours are slightly different (e.g., .jsonl vs .json)
DEV_FILE_PATH = os.path.join(PROJECT_ROOT, "task-dataset", "track_a", "gold-datasets", "eng_restaurant_gold_task3.json")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "subtask_3", "models", "best_model_restaurant.pt")
OUTPUT_IMAGE_PATH = os.path.join(SCRIPT_DIR, "poster_tsne_comparison.png")

BASE_MODEL_NAME = "roberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32


def load_data(f_path):
    print(f"Loading data from: {f_path}")
    if not os.path.exists(f_path):
        # Fallback: Check if it's a JSONL file instead of JSON
        f_path_l = f_path + "l"
        if os.path.exists(f_path_l):
            f_path = f_path_l
        else:
            print(f"❌ Error: File not found at {f_path}")
            return [], []

    texts = []
    labels = []

    with open(f_path, 'r', encoding='utf-8') as f:
        # Handle both JSON list and JSONL (line-by-line) formats
        try:
            data = json.load(f)  # Try standard JSON list
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]  # Fallback to JSONL

        for entry in data:
            # We need to iterate over the 'Quadruplet' list in the Gold file
            quads = entry.get('Quadruplet', [])

            for q in quads:
                aspect = q.get('Aspect')
                opinion = q.get('Opinion')
                category = q.get('Category')

                if aspect and category:
                    # Construct input exactly as the model expects: "Aspect [SEP] Opinion"
                    # This semantic context is what allows the model to cluster them
                    opinion_text = opinion if opinion else ""
                    text = f"{aspect} {opinion_text}"

                    texts.append(text)
                    labels.append(category)

        print(f"✅ Loaded {len(texts)} samples from {len(data)} sentences.")
        return texts, labels


def get_embeddings(texts, model, tokenizer):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
                DEVICE)
            outputs = model(**inputs)
            # Use CLS token (index 0)
            emb = outputs.last_hidden_state[:, 0, :]
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


def main():
    print("--- 🎨 Generating t-SNE Poster Visualization ---")

    # 1. Load Data
    texts, labels = load_data(DEV_FILE_PATH)
    if not texts:
        return

    # 2. Filter for Top Categories (Cleaner Plot)
    # We pick the top 8 most frequent categories to avoid a messy "rainbow" plot
    target_count = 14
    most_common = [k for k, v in Counter(labels).most_common(target_count)]

    indices = [i for i, x in enumerate(labels) if x in most_common]
    filtered_texts = [texts[i] for i in indices]
    filtered_labels = [labels[i] for i in indices]

    print(f"ℹ️  Filtering: Keeping top {target_count} categories ({len(filtered_texts)} samples).")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # 3. Generate "Before" Embeddings (Standard RoBERTa)
    print("\n[1/2] Computing Baseline Embeddings (RoBERTa-Base)...")
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME).to(DEVICE)
    emb_before = get_embeddings(filtered_texts, base_model, tokenizer)

    print("      Running t-SNE...")
    tsne_before = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto').fit_transform(
        emb_before)

    # 4. Generate "After" Embeddings (Your Trained Model)
    print("\n[2/2] Computing Trained Embeddings (Your Contrastive Model)...")
    trained_model = AutoModel.from_pretrained(BASE_MODEL_NAME).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        # Load the weights you trained
        trained_model.load_state_dict(torch.load(MODEL_PATH), strict=False)
    else:
        print(f"❌ Error: Trained model not found at {MODEL_PATH}")
        return

    emb_after = get_embeddings(filtered_texts, trained_model, tokenizer)

    print("      Running t-SNE...")
    tsne_after = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto').fit_transform(
        emb_after)

    # 5. Visualization
    print("\n--- 🖌️  Plotting ---")

    unique_labels = sorted(list(set(filtered_labels)))

    color_map = {
        # FOOD (Reds)
        'FOOD#QUALITY': '#e6194b',  # Red
        'FOOD#PRICES': '#f58231',  # Orange
        'FOOD#STYLE_OPTIONS': '#fabed4',  # Pink

        # DRINKS (Purples)
        'DRINKS#QUALITY': '#911eb4',  # Purple
        'DRINKS#STYLE_OPTIONS': '#e6beff',  # Lavender

        # SERVICE (Blues)
        'SERVICE#GENERAL': '#000075',  # Navy
        'SERVICE#QUALITY': '#4363d8',  # Medium Blue
        'SERVICE#PRICES': '#42d4f4',  # Cyan

        # AMBIENCE (Greens)
        'AMBIENCE#GENERAL': '#3cb44b',  # Green
        'AMBIENCE#STYLE_OPTIONS': '#aaffc3',  # Mint

        # RESTAURANT (Grays/Browns - Neutral)
        'RESTAURANT#GENERAL': '#800000',  # Maroon/Brown (Distinct from Location)
        'RESTAURANT#PRICES': '#a9a9a9',  # Dark Gray
        'RESTAURANT#STYLE_OPTIONS': '#000000',  # Black


        # LOCATION (Yellow/Gold - High Contrast vs Restaurant)
        'LOCATION#GENERAL': '#ffe119',  # Yellow/Gold
    }

    palette_dict = {}
    extra_colors = sns.color_palette("dark", n_colors=10)  # Backup colors
    for i, label in enumerate(unique_labels):
        if label in color_map:
            palette_dict[label] = color_map[label]
        else:
            palette_dict[label] = extra_colors[i % len(extra_colors)]

    sns.set_context("talk")  # Makes fonts bigger for posters
    palette = sns.color_palette("husl", len(unique_labels))

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # Plot Before
    sns.scatterplot(x=tsne_before[:, 0], y=tsne_before[:, 1],
                    hue=filtered_labels,
                    hue_order=unique_labels,
                    ax=axes[0],
                    palette=palette_dict,
                    s=180,
                    alpha=0.8,
                    legend=False,
                    edgecolor=None)
    axes[0].set_title("Before Training\n(Standard RoBERTa Embeddings)", fontsize=20, fontweight='bold', pad=20)
    axes[0].axis('off')

    # Plot After
    sns.scatterplot(x=tsne_after[:, 0], y=tsne_after[:, 1],
                    hue=filtered_labels,
                    hue_order=unique_labels,
                    ax=axes[1],
                    palette=palette_dict,
                    s=180,
                    alpha=0.85,
                    edgecolor=None)
    axes[1].set_title("After Contrastive Learning\n(Supervised SimCSE)", fontsize=20, fontweight='bold', pad=20)
    axes[1].axis('off')

    # Legend adjustments
    plt.legend(bbox_to_anchor=(1.05, 1),
               loc=2,
               borderaxespad=0.,
               title="Categories",
               fontsize=14,
               title_fontsize=16,
               frameon=False)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ Done! Plot saved to: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()