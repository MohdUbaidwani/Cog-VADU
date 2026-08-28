import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("ImageBind")
from ImageBind.imagebind import data
from ImageBind.imagebind.models.imagebind_model import ModalityType
from src.data.video_record import VideoRecord
from src.utils.sample_utils import uniform_temporal_subsample
from src.utils.torch_utils import initialize_vlm_model_and_device
from src.utils.path_utils import find_unprocessed_videos

def precompute_video_embeddings(
    video: VideoRecord,
    model,
    device,
    output_dir,
    batch_size=32,
    frame_interval=16,
    clip_duration=10,
    fps=30,
    num_samples=10,
    imagefile_template="{:06d}.jpg",
):
    """
    Precompute and save video embeddings for a single video.
    Saves as .npz file with embeddings and frame indices.
    """
    video_name = Path(video.path).name
    output_path = Path(output_dir) / f"{video_name}.npz"
    
    # Skip if already exists
    if output_path.exists():
        print(f"Skipping {video_name} - already exists")
        return
    
    frames_per_clip = int(clip_duration * fps)
    video_embeddings = []
    frame_indices = []
    
    for batch_start_frame in tqdm(
        range(0, video.num_frames, batch_size * frame_interval),
        desc=f"Processing {video_name}",
        unit="batch",
    ):
        batch_end_frame = min(
            batch_start_frame + (batch_size * frame_interval), 
            video.num_frames
        )
        batch_center_frame_idxs = list(range(
            batch_start_frame, batch_end_frame, frame_interval
        ))
        
        frame_indices.extend(batch_center_frame_idxs)
        
        # Prepare frame paths
        batch_clip_frame_paths = [
            [
                Path(video.path) / imagefile_template.format(frame_idx)
                for frame_idx in range(
                    max(clip_center_frame - frames_per_clip // 2, 0),
                    min(clip_center_frame + frames_per_clip // 2, video.num_frames),
                )
            ]
            for clip_center_frame in batch_center_frame_idxs
        ]
        
        # Subsample frames
        batch_clip_subsample_frame_paths = [
            uniform_temporal_subsample(clip_frame_paths, num_samples)
            for clip_frame_paths in batch_clip_frame_paths
        ]
        
        # Load and transform
        inputs = {
            ModalityType.VISION: data.load_and_transform_video_data(
                batch_clip_subsample_frame_paths, device
            ),
        }
        
        # Compute embeddings
        with torch.no_grad():
            embeddings = model(inputs)
            batch_embeddings = embeddings[ModalityType.VISION].cpu().numpy()
        
        video_embeddings.append(batch_embeddings)
    
    # Stack all embeddings
    all_embeddings = np.vstack(video_embeddings)
    
    # Compute norms
    norms = np.linalg.norm(all_embeddings, axis=1)
    
    # Save as compressed numpy file
    np.savez_compressed(
        output_path,
        embeddings=all_embeddings,
        frame_indices=np.array(frame_indices),
        norms=norms,
        video_name=video_name,
        num_frames=video.num_frames
    )
    
    print(f"Saved embeddings for {video_name}: {all_embeddings.shape}")
    print(f"  -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save precomputed embeddings")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--clip_duration", type=float, default=10)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--imagefile_template", type=str, default="{:06d}.jpg")
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
   
    
    args = parser.parse_args()
    
    # Initialize model once
    print("Initializing model...")
    model, device = initialize_vlm_model_and_device()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load video list
    video_list = [
        VideoRecord(x.strip().split(), args.root_path) 
        for x in open(args.annotationfile_path)
    ]
    
    # Split for parallel processing
    video_list = np.array_split(video_list, args.num_jobs)[args.job_index]
    
    

    print(f"Processing {len(video_list)} videos")
  
    
    # Process each video
    for video in video_list:
        try:
            precompute_video_embeddings(
                video=video,
                model=model,
                device=device,
                output_dir=output_dir,
                batch_size=args.batch_size,
                frame_interval=args.frame_interval,
                clip_duration=args.clip_duration,
                fps=args.fps,
                num_samples=args.num_samples,
                imagefile_template=args.imagefile_template,
              
            )
        except Exception as e:
            print(f"Error processing {video.path}: {e}")
            continue
    
    print("Done!")


if __name__ == "__main__":
    main()
