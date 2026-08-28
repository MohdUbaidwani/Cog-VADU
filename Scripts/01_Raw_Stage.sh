#!/bin/bash
#SBATCH --job-name="RAW1_stage"
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --partition=3090_risk
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/RAW1_stage_%A_%a.out
#SBATCH --error=output/RAW1_stage_%A_%a.err
export OMP_NUM_THREADS=8

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

dataset_dir="./ucf_crim_data"
batch_size=128  # Changed from 32 - VideoLLaMA3 is memory intensive
frame_interval=16
fps=30
fps_create=1.0  # FPS for creating template video clips
fps_vlama=1.0   # FPS for VideoLLaMA3 processing
T=10  # clip_duration in seconds
max_new_tokens=512  # Increased for better explanations


# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="path/to/video_filenames.txt"
output_scores_dir="path/to/RAW_STAGE1_scores/"

# Add resume and job control parameters
resume_flag="--resume"  # Add --resume if you want to resume processing
num_jobs=1
job_id=0  # Changed from job_index to job_id to match Python script

# Image file template
imagefile_template="{:06d}.jpg"
pathname="*.json"


source ~/miniconda3/etc/profile.d/conda.sh
conda activate vogvadu
# Create output directory
mkdir -p "$output_scores_dir"

echo "Configuration:"
echo "  Dataset dir: $dataset_dir"
echo "  Root path: $root_path"
echo "  Output dir: $output_scores_dir"
echo "  Batch size: $batch_size"
echo "  Frame interval: $frame_interval"
echo "  FPS: $fps (original)"
echo "  FPS create: $fps_create (for template videos)"
echo "  FPS vlama: $fps_vlama (for VideoLLaMA3)"
echo "  Clip duration: $T seconds"
echo "  Max new tokens: $max_new_tokens"
echo "  Image template: $imagefile_template"
echo ""

# Run the Python script with error handling
echo "Starting VideoLLaMA3 refinement..."
python -m MY_MODEL.Raw_CoADTP_STAGE1 \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_scores_dir "$output_scores_dir" \
    --fps "$fps" \
    --fps_create "$fps_create" \
    --fps_vlama "$fps_vlama" \
    --clip_duration "$T" \
    --imagefile_template "$imagefile_template" \
    --pathname "$pathname" \
    --max_new_tokens "$max_new_tokens" \
    --num_jobs "$num_jobs" \
    --job_id "$job_id" \
    $resume_flag

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Raw stage completed successfully!"
    echo "Results saved to: $output_scores_dir"
else
    echo "❌ Raw stage failed!"
    exit 1
fi
