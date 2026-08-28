import json
import os
import numpy as np
from sklearn.metrics import auc, roc_curve,precision_recall_curve

# Paths
#ann_root="Data/UCF_Eval.json"our_annotaion_ucf.json
ann_root = 'UCF_Eval.json'
print(ann_root,"this is th anootation file gt")
score_folder = './centred_STAGE2_COT36_refinedT3/refined_centered_score/'  # Your probabilistic .json scores (e.g., "0": 0.123)
print(score_folder)
# Load annotation
with open(ann_root, 'r', encoding='utf-8') as f:
    annotation = json.load(f)

all_predict_score = []
all_gt = []


import ipdb
for sample in annotation:
    key = sample['video'].split('/')[-1].split('.')[0]
    score_path = os.path.join(score_folder, key + '.json')
    
    if not os.path.exists(score_path):
        print(f"⚠️ Missing score file: {score_path}")
        continue

    with open(score_path, 'r') as f:
        segment_scores = json.load(f)

    # Make sure keys are in temporal order
    start_indices = sorted(segment_scores.keys(), key=lambda x: int(x))

    pred_score = []
    for i, start in enumerate(start_indices):###0,0,1,16,2,32...
        score = segment_scores[start]## take the segnment scoers 
        start = int(start)
        if i < len(start_indices) - 1:#
          
            next_start = int(start_indices[i + 1])
            num_frames = next_start - start
        else:###it loads all the scoers then multiplies by 16
            # Assume segment length of 16 for the last one (or customize as needed)
            num_frames = 16
        pred_score.extend([score] * num_frames)
    
    # Build ground truth
    video_len = sample['length']
    gt = [0.0] * video_len
    for i in range(0, len(sample['temporal_label']), 2):
        start = sample['temporal_label'][i]
        end = min(sample['temporal_label'][i+1], video_len)
        if start != -1:
            gt[start:end] = [1.0] * (end - start)

    # Align prediction and GT lengths
    pred_score = pred_score[:video_len]
    if len(pred_score) < video_len:
       
        pred_score += [0.0] * (video_len - len(pred_score))

    all_predict_score.extend(pred_score)
    all_gt.extend(gt)
  
   
    #import ipdb;ipdb.set_trace()    
print(len(all_gt),len(all_predict_score),"this is the length of all gt and ps")
# Compute metrics
fpr, tpr, _ = roc_curve(all_gt, all_predict_score)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(all_gt, all_predict_score)
ap = auc(recall, precision)


# Print results

print(f"✅ Final AUC Score (frame-level): {roc_auc:.4f}")
print(f"✅ Final AP Score (frame-level): {ap:.4f}")


