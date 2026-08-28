#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/05_create_summary_index_ucf_crime_%A_%a.out

# Set the UCF Crime directory
ucf_crime_dir="./ucf_crim_data"
export PYTHONPATH=$PYTHONPATH:./ImageBind

# Set paths
root_path="${ucf_crime_dir}/frames"
annotationfile_path="${ucf_crime_dir}/annotations/testss.txt"

batch_size=64
frame_interval=16
index_dim=1024

source ~/miniconda3/etc/profile.d/conda.sh
conda activate videollama3

captions_dir="${ucf_crime_dir}/captions/summary/raw_SUmmaries_FULL_UCF_cot36_TMR/"
output_dir="${ucf_crime_dir}/index/summary/Index_FULL_UCF_cot36_TMR_ours/"
python -m src.models.create_summary_index_FAST.py \
    --index_dim "$index_dim" \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --captions_dir "${captions_dir}" \
    --output_dir "${output_dir}"
