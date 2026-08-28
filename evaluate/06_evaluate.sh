#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/07_eval_ucf_crime_%A_%a.out

# Set the UCF Crime directory
ucf_crime_dir=""


root_path="/path to extracted frames/"
annotationfile_path="/video_filenames/"            #EXP_ref
temporal_annotation_file="path /to/actual/annotation.txt"

frame_interval=16
num_neighbors=10
normal_label=7
video_fps=30

# Evaluate the AUC-ROC of clip-level scores after anomaly score refinement
captions_dir="path/to/clean_summary"
#TEST3new_blip/"
echo $captions_dir

scores_dir="path/to/store/central_refined_scores/"


similarity_dir="path/to/similarity_dir"

output_dir="path/to/refined_scores_dir/"

output_base="path/to/base_dir/to_store/central_scores"
# Create base output dir
mkdir -p "$output_base"

# === ABLATION LOOP: num_neighbors from 2 to 10 ===
for nn in {2..10}; do
    echo "============================================"
    echo "Running with num_neighbors = $nn"
    echo "============================================"

    output_dir="${output_base}/nn_${nn}"
    mkdir -p "$output_dir"

    python -m src.eval2_k_exp \
        --root_path "$root_path" \
        --annotationfile_path "$annotationfile_path" \
        --temporal_annotation_file "$temporal_annotation_file" \
        --scores_dir "$scores_dir" \
        --similarity_dir "$similarity_dir" \
        --captions_dir "$captions_dir" \
        --output_dir "$output_dir" \
        --frame_interval "$frame_interval" \
        --normal_label "$normal_label" \
        --num_neighbors "$nn" \
        --video_fps "$video_fps"

    # Extract AUCs
    roc=$(cat "$output_dir/roc_auc_nn_${nn}.txt")
    pr=$(cat "$output_dir/pr_auc_nn_${nn}.txt")
    echo "→ ROC-AUC: $roc  |  PR-AUC: $pr"
    echo
done

echo "Ablation complete! Results in: $output_base"
