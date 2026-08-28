#!/bin/bash
videos_dir=""
dataset_dir=""
# Set paths

frames_dir="${dataset_dir}/frames"
annotations_file="${dataset_dir}/annotations/testss.txt"

python src/preprocessing/extract_frames.py \
    --videos_dir "$videos_dir" \
    --frames_dir "$frames_dir" \
    --annotations_file "$annotations_file"
