import matplotlib
matplotlib.use('Agg')  # Redundant but ensures local fallback
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os  # For access checks

def plot_scores(scores, labels, video_name, save_dir, normal_id=7):
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanity: Bail early if no data
    if len(scores) == 0:
        print(f"[WARN] Skipping plot for {video_name}: empty scores")
        return
    
    fig, ax = plt.subplots(figsize=(18, 4))
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.06, right=0.99)

    x = np.arange(scores.shape[0])
    ax.plot(x, scores, color="#4e79a7", linewidth=1)
    
    # Dynamic ylims if scores aren't normalized
    if scores.max() > 1 or scores.min() < 0:
        buffer = (scores.max() - scores.min()) * 0.1
        ymin, ymax = scores.min() - buffer, scores.max() + buffer
    else:
        ymin, ymax = 0, 1
    xmin, xmax = 0, scores.shape[0]
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # Title escaping (your code, with % fixed if using LaTeX math mode)
    title = video_name
    title = title.replace("#", r"\#")
    title = title.replace("%", r"\%")
    title = title.replace("_", r"\_")
    title = title.replace("&", r"\&")
    title = title.replace("{", r"\{")
    title = title.replace("}", r"\}")
    title = title.replace("^", r"\^{}")

    # Red rectangles for anomaly regions
    start_idx = None
    for i in range(labels.shape[0]):
        if labels[i] != normal_id and start_idx is None:
            start_idx = i
        elif labels[i] == normal_id and start_idx is not None:
            rect = plt.Rectangle(
                (start_idx, ymin),
                i - start_idx,
                ymax - ymin,
                color="#e15759",
                alpha=0.5,
            )
            ax.add_patch(rect)
            start_idx = None
    if start_idx is not None:
        rect = plt.Rectangle(
            (start_idx, ymin),
            labels.shape[0] - start_idx,
            ymax - ymin,
            color="#e15759",
            alpha=0.5,
        )
        ax.add_patch(rect)

    ax.text(0.02, 0.90, title, fontsize=28, transform=ax.transAxes)
    for yline in np.linspace(ymin, ymax, 4)[1:-1]:  # Dynamic lines based on ylims
        ax.axhline(y=yline, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(np.linspace(ymin, ymax, 5)[1:-1])  # Matching ticks
    ax.tick_params(axis="y", labelsize=28)
    ax.tick_params(axis="x", labelsize=28)

    ax.set_ylabel("Anomaly score", fontsize=28)
    ax.set_xlabel("Frame number", fontsize=28)

    fig_file = save_dir / f"{video_name}_scores.png"
    try:
        plt.savefig(fig_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[SAVED] Score plot: {fig_file}")
        print(f"[DEBUG] File size: {fig_file.stat().st_size if fig_file.exists() else 0} bytes")
    except Exception as e:
        print(f"[ERROR] Failed to save {fig_file}: {e}")
        plt.close(fig)  # Clean up anyway