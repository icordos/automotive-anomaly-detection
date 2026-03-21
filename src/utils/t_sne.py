"""
This code contains a helper function used to plot t-SNE in order to
perform in depth error analysis.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import json
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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


def plot_tsne_3d(npz_path: str, out_path: str = None):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]   # (N, 1536)
    labels     = data["labels"]       # 0=good, 1=bad
    scores     = data["scores"]       # anomaly scores
    paths      = data["paths"]

    print(f"Running 3D t-SNE on {len(labels)} images ({labels.sum()} bad, {(labels==0).sum()} good)...")
    tsne = TSNE(
        n_components=3,
        perplexity=min(30, max(5, len(labels) // 4)),
        random_state=42,
        n_iter=1000,
        n_jobs=-1,
    )
    coords = tsne.fit_transform(embeddings)   # (N, 3)

    good_mask = labels == 0
    bad_mask  = labels == 1

    # Normalize scores to [0,1] for marker size
    s_min, s_max = scores.min(), scores.max()
    norm_scores = (scores - s_min) / (s_max - s_min + 1e-12)

    fig = go.Figure()

    # Good samples
    fig.add_trace(go.Scatter3d(
        x=coords[good_mask, 0],
        y=coords[good_mask, 1],
        z=coords[good_mask, 2],
        mode="markers",
        name=f"Good ({good_mask.sum()})",
        marker=dict(
            size=4,
            color="steelblue",
            opacity=0.7,
        ),
        customdata=np.stack([
            scores[good_mask],
            [Path(p).name for p in paths[good_mask]],
        ], axis=1),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Score: %{customdata[0]:.4f}<br>"
            "x: %{x:.2f}, y: %{y:.2f}, z: %{z:.2f}"
            "<extra></extra>"
        ),
    ))

    # Bad samples — sized by anomaly score so the worst offenders are largest
    fig.add_trace(go.Scatter3d(
        x=coords[bad_mask, 0],
        y=coords[bad_mask, 1],
        z=coords[bad_mask, 2],
        mode="markers",
        name=f"Bad ({bad_mask.sum()})",
        marker=dict(
            size=5 + norm_scores[bad_mask] * 8,   # 5–13px based on score
            color="crimson",
            opacity=0.9,
            line=dict(color="black", width=0.5),
        ),
        customdata=np.stack([
            scores[bad_mask],
            [Path(p).name for p in paths[bad_mask]],
        ], axis=1),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Score: %{customdata[0]:.4f}<br>"
            "x: %{x:.2f}, y: %{y:.2f}, z: %{z:.2f}"
            "<extra></extra>"
        ),
    ))

    category = Path(npz_path).stem.replace("_tsne_data", "").split("_", 2)[-1]

    fig.update_layout(
        title=dict(text=f"t-SNE 3D — {category}"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        scene=dict(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            zaxis_title="Dim 3",
        ),
    )

    out_html = out_path or npz_path.replace(".npz", "_3d.html")
    out_png  = (out_path or npz_path.replace(".npz", "_3d.png")).replace(".html", ".png")

    # Save interactive HTML (best for exploration)
    fig.write_html(out_html)
    print(f"Saved interactive: {out_html}")

    # Save static PNG snapshot
    fig.write_image(out_png)
    print(f"Saved static:      {out_png}")

    # Metadata sidecar
    with open(out_png + ".meta.json", "w") as f:
        json.dump({
            "caption": f"t-SNE 3D — {category}",
            "description": f"3D t-SNE of {len(labels)} test images. Good={good_mask.sum()}, Bad={bad_mask.sum()}. Bad markers sized by anomaly score."
        }, f)

    return coords, labels, scores


## Usage for 2D:
# plot_tsne(r"C:\Users\victo\Downloads\artifacts\artifacts\clients\client_1\client_1_pipe_clip_tsne_data.npz")
# plot_tsne(r"C:\Users\victo\Downloads\artifacts\artifacts\clients\client_1\client_1_engine_wiring_tsne_data.npz")

## Usage for 3D:
# plot_tsne_3d(r"C:\Users\victo\Downloads\artifacts\artifacts\clients\client_1\client_1_pipe_clip_tsne_data.npz")
# plot_tsne_3d(r"C:\Users\victo\Downloads\artifacts\artifacts\clients\client_1\client_1_engine_wiring_tsne_data.npz")
# plot_tsne_3d(r"C:\Users\victo\Downloads\artifacts\artifacts\clients\client_3\client_3_underbody_screw_tsne_data.npz")
# plot_tsne_3d(r"C:\Users\victo\Downloads\artifacts\artifacts\clients\client_3\client_3_underbody_pipes_tsne_data.npz")
# plot_tsne_3d(r"C:\Users\victo\Downloads\artifacts\artifacts\clients\client_2\client_2_tank_screw_tsne_data.npz")
plot_tsne_3d(r"C:\Users\victo\Downloads\artifacts\artifacts\clients\client_2\client_2_pipe_staple_tsne_data.npz")



