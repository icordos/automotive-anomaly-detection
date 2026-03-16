"""
This code contains a helper function used to plot t-SNE in order to
perform in depth error analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

def plot_tsne(npz_path: str, out_path: str = None):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]   # (N, 1536)
    labels     = data["labels"]       # 0=good, 1=bad
    scores     = data["scores"]       # anomaly scores for color intensity
    paths      = data["paths"]

    print(f"Running t-SNE on {len(labels)} images ({labels.sum()} bad, {(labels==0).sum()} good)...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(labels)//4), random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeddings)   # (N, 2)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    good_mask = labels == 0
    bad_mask  = labels == 1

    ax.scatter(coords[good_mask, 0], coords[good_mask, 1],
               c="steelblue", alpha=0.6, s=40, label=f"Good ({good_mask.sum()})")
    ax.scatter(coords[bad_mask,  0], coords[bad_mask,  1],
               c="crimson",   alpha=0.8, s=60, label=f"Bad ({bad_mask.sum()})",
               edgecolors="black", linewidths=0.5)

    ax.legend(fontsize=12)
    ax.set_title(Path(npz_path).stem.replace("_tsne_data", ""), fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    out = out_path or npz_path.replace(".npz", "_plot.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

# Usage:
# plot_tsne("artifacts/federated_sequential/client_1/client_1_categoryX_tsne_data.npz")
