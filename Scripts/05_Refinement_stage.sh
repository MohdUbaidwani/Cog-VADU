#!/bin/bash
#SBATCH --job-name="refX1org"
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --partition=3090_risk
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/04_query_llm_ucf_crime_%A_%a.out
#SBATCH --error=output/04_query_llm_ucf_crime_%A_%a.err
export OMP_NUM_THREADS=8

################6_FAST_REF_ucf_exps_TMR.sh
dataset_dir="path/dataset/dir/"

batch_size=32
frame_interval=16
fps=30  # Change this to the frame rate of your videos
T=10
N=10
num_neighbors=10


source ~/miniconda3/etc/profile.d/conda.sh
conda activate cogvadu

# Set paths
root_path="${dataset_dir}/frames"
annotationfile_path="path/to/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
   

captions_dir="path to raw_summaries directory/"
index_dir="path to summary Index Directory"
embeddings_dir="path to Visual embeddings directory"
scores_dir="path to raw_scores directory"
output_scores_dir="output path to refined_scores directory"
output_summary_dir="output path to refined_summaries directory"
output_similarity_dir="output path to semantic similarity directory"
output_filenames_dir="output path to refined_filenames directory"
resume_flag="--resume"   
# Run the Python script with the specified parameters
python -m src.models.video_text_TEST3_fast \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --video_embeddings_dir "$embeddings_dir"\
    --output_scores_dir "$output_scores_dir" \
    --output_summary_dir "$output_summary_dir" \
    --output_similarity_dir "$output_similarity_dir" \
    --output_filenames_dir "$output_filenames_dir" \
    --captions_dir "$captions_dir" \
    --index_dir "$index_dir" \
    --scores_dir "$scores_dir" \
    --fps "$fps" \
    --clip_duration "$T" \
    --num_samples "$N" \
    --num_neighbors "$num_neighbors"
    #$resume_flag
