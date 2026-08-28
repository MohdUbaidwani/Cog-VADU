#!/bin/bash
#SBATCH --job-name="get_embeddings"
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --partition=3090_risk
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/embeddings_%A_%a.out
#SBATCH --error=output/embeddings_%A_%a.err

# Set environment variables
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Configuration parameters
dataset_dir="./ucf_crim_data"
batch_size=128
frame_interval=16
fps=30
T=10

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="path/to/video_filenames.txt" #testss.txt
output_embeddings_dir="path/to/Embedding_dir/"
num_jobs=1
job_index=0

# Image file template
imagefile_template="{:06d}.jpg"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate videollama3


# Create output directory
mkdir -p "$output_embeddings_dir"

echo "Configuration:"
echo "  Dataset dir: $dataset_dir"
echo "  Root path: $root_path"
echo "  Batch size: $batch_size"
echo "  Frame interval: $frame_interval"
echo "  FPS: $fps"
echo "  Clip duration: $T seconds"
echo "  Image template: $imagefile_template"
echo "  Output embeddings: $output_embeddings_dir"


# Verify imports
python -c "import torch; import torchaudio; print('✓ All imports successful')" || exit 1

python -m src.models.get_visual_embeddings \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --output_dir "$output_embeddings_dir" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --fps "$fps" \
    --clip_duration "$T" \
    --imagefile_template "$imagefile_template" \
    --num_jobs "$num_jobs" \
    --job_index "$job_index"
