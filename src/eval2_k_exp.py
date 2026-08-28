#!/usr/bin/env python3
# eval_ablation.py

import argparse
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# === CUSTOM IMPORTS (adjust if needed) ===
from src.data.video_record import VideoRecord
from src.utils.vis_utils import visualize_video


# ------------------------------------------------------------
# 1. Load temporal annotations
# ------------------------------------------------------------
def load_temporal_annotations(file_path):
    annotations = {}
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            video_name = Path(parts[0]).stem
            values = parts[2:]
            annotations[video_name] = [v for v in values if v != "-1"]
    return annotations


# ------------------------------------------------------------
# 2. Generate frame-level labels
# ------------------------------------------------------------
def get_video_labels(video: VideoRecord, annotations, normal_label):
    video_name = Path(video.path).stem
    if video_name not in annotations:
        return [normal_label] * video.num_frames

    intervals = annotations[video_name]
    starts = intervals[::2]
    ends = intervals[1::2]

    labels = []
    for frame_idx in range(video.num_frames):
        global_idx = frame_idx + video.start_frame
        label = normal_label
        for s, e in zip(starts, ends):
            if int(s) <= global_idx <= int(e):
                label = 1  # anomaly
                break
        labels.append(label)
    return labels


# ------------------------------------------------------------
# 3. Weighted scoring with TOP-K neighbors
# ------------------------------------------------------------
def calculate_weighted_scores(scores_dict, similarity_dict, num_neighbors, frame_interval):
    scores = []

    for frame_idx in scores_dict.keys():
        fidx = str(frame_idx)

        # CASE 1: Nested scores → use neighbor weighting
        if isinstance(scores_dict[frame_idx], dict):
            neigh_scores = scores_dict[fidx]
            neigh_sim = similarity_dict.get(fidx, {})

            # Get common neighbors
            common = set(neigh_scores.keys()) & set(neigh_sim.keys())
            if not common:
                final_score = float(np.mean(list(neigh_scores.values())))
            else:
                s_vals = np.array([neigh_scores[k] for k in common])
                sim_vals = np.array([neigh_sim[k] for k in common])

                # TOP-K by similarity
                if len(s_vals) > num_neighbors:
                    topk_idx = np.argsort(sim_vals)[-num_neighbors:]
                    s_vals = s_vals[topk_idx]
                    sim_vals = sim_vals[topk_idx]

                weights = np.exp(sim_vals) / np.sum(np.exp(sim_vals))
                final_score = np.sum(s_vals * weights)

        # CASE 2: Flat score → multiply by average similarity
        else:
            base_score = scores_dict[frame_idx]
            if fidx in similarity_dict and isinstance(similarity_dict[fidx], dict):
                sims = np.array(list(similarity_dict[fidx].values())[:num_neighbors])
                conf = np.mean(sims)
                final_score = base_score * conf
            else:
                final_score = base_score

        scores.append(final_score)

    # Repeat each center score for `frame_interval` frames
    full_scores = np.repeat(scores, frame_interval)
    return full_scores, scores  # full, centered


# ------------------------------------------------------------
# 4. Save metric
# ------------------------------------------------------------
def save_metric(output_dir, name, nn, value):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{name}_nn_{nn}.txt", "w") as f:
        f.write(f"{value:.6f}\n")


# ------------------------------------------------------------
# 5. Main evaluation
# ------------------------------------------------------------
def main(args):
    # Paths
    scores_dir = Path(args.scores_dir)
    sim_dir = Path(args.similarity_dir)
    cap_dir = Path(args.captions_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    annotations = load_temporal_annotations(args.temporal_annotation_file)

    # Load video list
    video_list = [
        VideoRecord(line.strip().split(), args.root_path)
        for line in open(args.annotationfile_path)
        if line.strip()
    ]
    print(f"Processing {len(video_list)} videos")

    all_scores = []
    all_labels = []

    for video in video_list:
        vname = Path(video.path).name
        vstem = Path(video.path).stem

        # Load JSONs
        score_path = scores_dir / f"{vname}.json"
        sim_path = sim_dir / f"{vname}.json"
        cap_path = cap_dir / f"{vname}.json"

        if not score_path.exists():
            print(f"Missing scores: {score_path}")
            continue

        with open(score_path) as f:
            scores_dict = json.load(f)
        with open(sim_path) as f:
            sim_dict = json.load(f)
        with open(cap_path) as f:
            captions = json.load(f)

        # Labels
        labels = get_video_labels(video, annotations, args.normal_label)

        # Weighted scores
        full_scores, center_scores = calculate_weighted_scores(
            scores_dict, sim_dict, args.num_neighbors, args.frame_interval
        )
        full_scores = full_scores[: video.num_frames]  # trim

        # Save centered scores
        center_dict = {str(i * args.frame_interval): s for i, s in enumerate(center_scores)}
        (out_dir / "refined_centered_score").mkdir(exist_ok=True)
        with open(out_dir / "refined_centered_score" / f"{vname}.json", "w") as f:
            json.dump(center_dict, f, indent=2)

        # Collect
        all_scores.extend(full_scores)
        all_labels.extend(labels)

        # Visualize
        if args.visualize:
            visualize_video(
                vname, [], full_scores, captions, video.path,
                args.video_fps, out_dir / f"{vstem}_vis.mp4",
                args.normal_label, "{:06d}.jpg", None
            )

    # Metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    binary_labels = (all_labels != args.normal_label).astype(float)

    fpr, tpr, _ = roc_curve(binary_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    save_metric(out_dir, "roc_auc", args.num_neighbors, roc_auc)

    precision, recall, _ = precision_recall_curve(binary_labels, all_scores)
    pr_auc = auc(recall, precision)
    save_metric(out_dir, "pr_auc", args.num_neighbors, pr_auc)

    print(f"\nRESULTS (nn={args.num_neighbors})")
    print(f"ROC-AUC : {roc_auc:.6f}")
    print(f"PR-AUC  : {pr_auc:.6f}")
    print(f"Frames  : {len(all_scores)}")
    print(f"Saved to: {out_dir}")


# ------------------------------------------------------------
# 6. CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly Detection Evaluation + Ablation")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--temporal_annotation_file", type=str, required=True)
    parser.add_argument("--scores_dir", type=str, required=True)
    parser.add_argument("--similarity_dir", type=str, required=True)
    parser.add_argument("--captions_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--normal_label", type=int, default=0)
    parser.add_argument("--num_neighbors", type=int, default=2)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--video_fps", type=float)

    args = parser.parse_args()
    if args.visualize and args.video_fps is None:
        parser.error("--video_fps required with --visualize")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
