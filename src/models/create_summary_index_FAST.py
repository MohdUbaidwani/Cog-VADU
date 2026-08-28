
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List

import faiss
import torch
from tqdm import tqdm

# ----------------------------------------------------------------------
# Your project imports
# ----------------------------------------------------------------------
from src.data.video_record import VideoRecord
from src.utils.torch_utils import initialize_vlm_model_and_device

# ImageBind
import sys
sys.path.append("libs/ImageBind")
from ImageBind.imagebind import data
from ImageBind.imagebind.models.imagebind_model import ModalityType


# ----------------------------------------------------------------------
# Argparse
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Build deduplicated FAISS text index")
    parser.add_argument("--index_dim", type=int, default=1024)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--captions_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_video_records(annotationfile_path: str, root_path: str) -> List[VideoRecord]:
    with open(annotationfile_path) as f:
        lines = f.readlines()
    return [VideoRecord(line.strip().split(), root_path) for line in lines]


def initialize_faiss_index(dim: int) -> faiss.Index:
    """Inner-product index (vectors are L2-normalized before add)."""
    return faiss.IndexFlatIP(dim)


def embed_text(model, device, texts: List[str]) -> torch.Tensor:
    inputs = {ModalityType.TEXT: data.load_and_transform_text(texts, device)}
    with torch.no_grad():
        emb = model(inputs)[ModalityType.TEXT]          # (B, D)
    return emb


def dedup_batch(
    batch_frame_idxs,
    caption_to_frames: dict[str, List[int]],
    video_captions: dict[str, str],
) -> List[int]:
    """
    Return only the *earliest* frame for each unique caption in the batch.
    """
    seen = set()
    keepers = []
    for fidx in batch_frame_idxs:
        cap = video_captions[str(fidx)]
        if cap in seen:
            continue
        # Earliest frame that owns this caption
        earliest = min(caption_to_frames[cap])
        if fidx == earliest:
            keepers.append(fidx)
            seen.add(cap)
    return keepers


def save_results(
    index: faiss.Index,
    file_names: List[str],
    idx_to_all_frames: dict[int, List[int]],
    out_dir: Path,
    video_name: str,
):
    # 1. FAISS binary
    faiss.write_index(index, str(out_dir / f"{video_name}.bin"))

    # 2. Representative file names (one per unique caption)
    with open(out_dir / f"{video_name}.json", "w") as f:
        json.dump(file_names, f)

    # 3. Mapping FAISS idx → *all* frames that share the caption
    with open(out_dir / f"{video_name}_idx2frames.json", "w") as f:
        json.dump(idx_to_all_frames, f)


# ----------------------------------------------------------------------
# Core per-video processing
# ----------------------------------------------------------------------
def process_video(
    video: VideoRecord,
    model,
    device,
    dim: int,
    batch_size: int,
    frame_interval: int,
    captions_dir: Path,
    out_dir: Path,
):
    video_name = Path(video.path).name
    index = initialize_faiss_index(dim)
    file_names = []                 # one entry per *unique* caption
    idx_to_all_frames = {}          # FAISS idx → list of all frame indices

    # ------------------------------------------------------------------
    # Load captions + build reverse mapping
    # ------------------------------------------------------------------
    cap_path = captions_dir / f"{video_name}.json"
    with open(cap_path) as f:
        video_captions = json.load(f)          # { "0": "caption …", "16": … }

    caption_to_frames = defaultdict(list)
    for fidx_str, cap in video_captions.items():
        fidx = int(fidx_str)
        caption_to_frames[cap].append(fidx)

    # ------------------------------------------------------------------
    # Iterate over the video in batches
    # ------------------------------------------------------------------
    for start in tqdm(
        range(0, video.num_frames, batch_size * frame_interval),
        desc=f"Indexing {video.path}",
        unit="batch",
    ):
        end = min(start + batch_size * frame_interval, video.num_frames)
        batch_idxs = range(start, end, frame_interval)

        # ---- deduplicate inside this batch ----
        batch_idxs = dedup_batch(batch_idxs, caption_to_frames, video_captions)

        if not batch_idxs:
            continue

        try:
            texts = [video_captions[str(fi)] for fi in batch_idxs]
        except KeyError:
            print("❌ Missing caption index:", fi)
            print("❌ Caption file:", caption_file_path)
            print("❌ Video path:", video_path if 'video_path' in locals() else "N/A")
            raise
     	

        # ---- embed & add to FAISS ----
        emb = embed_text(model, device, texts)                # (B, D)
        vecs = emb.cpu().numpy()
        faiss.normalize_L2(vecs)
        index.add(vecs)                                       # adds B vectors

        # ---- bookkeeping ----
        cur_start_idx = len(file_names)
        for local_i, fi in enumerate(batch_idxs):
            faiss_idx = cur_start_idx + local_i
            file_names.append(f"{video_name}/{fi}")
            idx_to_all_frames[faiss_idx] = caption_to_frames[video_captions[str(fi)]]

    # ------------------------------------------------------------------
    # Persist everything
    # ------------------------------------------------------------------
    save_results(index, file_names, idx_to_all_frames, out_dir, video_name)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, device = initialize_vlm_model_and_device()

    videos = load_video_records(args.annotationfile_path, args.root_path)

    for vid in videos:
        process_video(
            video=vid,
            model=model,
            device=device,
            dim=args.index_dim,
            batch_size=args.batch_size,
            frame_interval=args.frame_interval,
            captions_dir=Path(args.captions_dir),
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
