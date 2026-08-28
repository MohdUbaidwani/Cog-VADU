#!/bin/bash

Raw_stage_folder=""
output_raw_scores_dir=""
output_raw_summary_dir=""

python src/preprocessing/extract_raw_scores_summaries.py \
    --Raw_stage_folder "$Raw_stage_folder" \
    --output_raw_scores_dir "$output_raw_scores_dir" \
    --output_raw_summary_dir "$output_raw_summary_dir"
