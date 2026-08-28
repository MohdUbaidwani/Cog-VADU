import argparse
import json
from pathlib import Path
import ipdb
import os
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve,roc_auc_score

from src.data.video_record import VideoRecord
from src.utils.vis_utils import visualize_video
def temporal_testing_annotations(temporal_annotation_file):
    annotations = {}

    with open(temporal_annotation_file) as annotations_f:
        for line in annotations_f:
            parts = line.strip().split()
            video_name = str(Path(parts[0]).stem)
            annotation_values = parts[2:]
            annotations[video_name] = annotation_values

    return annotations


def get_video_labels(video_record, annotations, normal_label):
    video_name = Path(video_record.path).name
    labels = []

    video_annotations = [x for x in annotations[video_name] if x != "-1"]

    # Separate start and stop indices
    start_indices = video_annotations[::2]
    stop_indices = video_annotations[1::2]

    for frame_index in range(video_record.num_frames):
        frame_label = normal_label

        # Check if the current frame index falls within any annotation range
        if len(video_record.label) == 1:
            for start_idx, end_idx, label in zip(
                start_indices, stop_indices, video_record.label * len(start_indices)
            ):
                if int(start_idx) <= frame_index + video_record.start_frame <= int(end_idx):
                    frame_label = label
        else:
            video_labels = video_record.label

            # Pad video_labels if it's shorter than start_indices
            if len(video_labels) < len(start_indices):
                last_label = [video_record.label[-1]] * (len(start_indices) - len(video_labels))
                video_labels.extend(last_label)

            for start_idx, end_idx, label in zip(start_indices, stop_indices, video_labels):
                if int(start_idx) <= frame_index + video_record.start_frame <= int(end_idx):
                    frame_label = label

        labels.append(frame_label)
    #ipdb.set_trace()
    return labels


def calculate_weighted_scores(scores_dict, similarity_dict, num_neighbors, frame_interval):
    scores = []
    for frame_idx in scores_dict.keys():
        # check if scores_dict is a dict of dicts
        if isinstance(scores_dict[frame_idx], dict):#False scores_dict["0"]
#{'0': 0.3, '1': 0.3, '2': 0.2, '3': 0.2, '4': 0.3, '5': 0.3, '6': 0.3, '7': 0.3, '8': 0.3, '9': 0.3}
            str_idx=str(frame_idx)
            available_neighbours=list(scores_dict[str_idx].keys())
            frame_scores = np.array(
                [scores_dict[str_idx][str(nn_idx)] for nn_idx in (available_neighbours)]
            )#ipdb> frame_scores
#array([0.3, 0.#3, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])

            frame_similarity = np.array(
                [similarity_dict[str_idx][str(nn_idx)] for nn_idx in (available_neighbours)]
            )#array([0.3617309 , 0.35894465, 0.35802969, 0.35546637, 0.35006663,
      # 0.34442052, 0.3432368 , 0.34228387, 0.33930212, 0.3392303 ])

            frame_weights = np.exp(frame_similarity) / np.sum(np.exp(frame_similarity))
 #           array([0.3617309 , 0.35894465, 0.35802969, 0.35546637, 0.35006663,
#       0.34442052, 0.3432368 , 0.34228387, 0.33930212, 0.3392303 ])
          
            scores.append(np.sum(frame_scores * frame_weights))# np.sum(frame_scores*frame_weights)=0.27
            #import ipdb; ipdb.set_trace()
        else:
            scores.append(scores_dict[frame_idx])
    centerd_scores=scores
    scores = np.repeat(scores, frame_interval)

    return scores,centerd_scores


def save_metric(output_dir, metric_name, num_neighbors, metric_value):
    with open(output_dir / f"{metric_name}_nn_{num_neighbors}.txt", "w") as f:
        f.write(f"{metric_value}\n")


def main(
    root_path,
    annotationfile_path,
    temporal_annotation_file,
    scores_dir,
    similarity_dir,
    captions_dir,
    output_dir,
    frame_interval,
    normal_label,
    num_neighbors,
    without_labels,
    visualize,
    video_fps,
):
    # Convert paths to Path objects
    scores_dir = Path(scores_dir)
    similarity_dir = Path(similarity_dir)
    captions_dir = Path(captions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the temporal annotations
    if not without_labels:
        annotations = temporal_testing_annotations(temporal_annotation_file)
    with open (temporal_annotation_file,"r") as f:
        annotation_data=[line.strip().split() for line in f]
    video_to_class={Path(parts[0]).stem: parts[1] for parts in annotation_data}
    
    # Load video records from the annotation file
    video_list = [VideoRecord(x.strip().split(), root_path) for x in open(annotationfile_path) if x.strip()]
    print(len(video_list),"this si the ;ength of viodeos being provesesd")
    
   # print(video_list)

    flat_scores = []
    flat_labels = []
    class_results={
    	"Normal_Videos_":[],
    	"Abuse":[],
    	"Explosion":[],
    	"Robbery":[],
    	"Burglary":[],
    	"Assault":[],
    	"Vandalism":[],
    	"Stealing":[],
    	"RoadAccidents":[],
    	"Arrest":[],
    	"Shooting":[],
    	"Fighting":[],
    	"Shoplifting":[],
    	"Arson":[],
    
    }
    
    for video in video_list:
        video_name = Path(video.path).name
        print(scores_dir,"tbibdhubudvwuegtvduqb	ankl")#cf_crim_data/scores/refined/llama-2-13b-chat/opt-6.7b-coco+opt-6.7b+flan-#t5-xxl+flan-t5-xl+flan-t5-xl-coco/2035605_002_if_you_wer tbibdhubudvwuegtvduqb
        # Load the scores and similarity
        video_scores_path = scores_dir / f"{video_name}.json"
       
        video_similarity_path = similarity_dir / f"{video_name}.json"
        video_captions_path = captions_dir / f"{video_name}.json"
        print(video_captions_path)
        with open(video_scores_path) as f:
            video_scores_dict = json.load(f)
        #import ipdb; ipdb.set_trace()
        
        with open(video_similarity_path) as f:
            video_similarity = json.load(f)
        
        with open(video_captions_path) as f:
            video_captions = json.load(f)

        # Get video labels
        if without_labels:
            video_labels = []
        else:
            video_labels = get_video_labels(video, annotations, normal_label)
        
        video_scores,centerd_scores = calculate_weighted_scores(
            video_scores_dict, video_similarity, num_neighbors, frame_interval
        )
        centerd_scores_dict={str(i*frame_interval):score for i ,score in enumerate(centerd_scores)}
        video_name=os.path.basename(video.path)
        # Define subdirectory to store centered score JSONs
        refined_score_save_dir = output_dir / "refined_centered_score"
        refined_score_save_dir.mkdir(parents=True, exist_ok=True)

# Define the full path for this video's JSON file
        output_score_path = refined_score_save_dir / f"{video_name}.json"
        with open(output_score_path,"w") as f:
            json.dump(centerd_scores_dict, f,indent=4)
        
       # import ipdb; ipdb.set_trace()  
        video_scores = video_scores[: video.num_frames]
        ##per class auc ap
        video_class=video_to_class.get(Path(video.path).stem,"Unknown")
        if not without_labels and any(label!=normal_label for label in video_labels):
            video_binary_labels=[1 if label!= normal_label else 0 for label in video_labels]
            try:
                auc_score = roc_auc_score(video_binary_labels, video_scores)
                precision,recall,_=precision_recall_curve(video_binary_labels ,video_scores)
                ap_score=auc(recall,precision)
                class_results[video_class].append({
                "video_id":video_name,
                "auc":auc_score,
                "ap":ap_score,
                "scores":video_scores,
                "labels":video_binary_labels})
            except ValueError:
                print(f" skipping auc/ap fkor {video_name} : nott enough positive or neg samples")
        # Extend scores and labels
        flat_scores.extend(video_scores)
        if not without_labels:
            flat_labels.extend(video_labels)

        if visualize:
            # visualize_video
            visualize_video(
                video_name,
                [],
                video_scores,
                video_captions,
                video.path,
                video_fps,
                output_dir / f"{video_name}.mp4",
                normal_label,
                "{:06d}.jpg",
                None,
            )

    flat_scores = np.array(flat_scores)

    if not without_labels:
            flat_labels = np.array(flat_labels)
            flat_binary_labels = flat_labels != normal_label

            # Compute overall ROC AUC score
            try:
                roc_auc = roc_auc_score(flat_binary_labels, flat_scores)
                save_metric(output_dir, "roc_auc", num_neighbors, roc_auc)
                print(f"Overall AUC: {roc_auc:.4f}")
            except ValueError as e:
                print(f"Error computing overall AUC: {e}")

            # Compute overall precision-recall curve
            try:
                precision, recall, _ = precision_recall_curve(flat_binary_labels, flat_scores)
                pr_auc = auc(recall, precision)
                print(f"Overall AP: {pr_auc:.4f}, GT length: {len(flat_binary_labels)}, Pred length: {len(flat_scores)}")
                save_metric(output_dir, "pr_auc", num_neighbors, pr_auc)
            except ValueError as e:
                print(f"Error computing overall AP: {e}")

            # Compute and save class-wise AUC/AP
            for class_name, class_videos in class_results.items():
                if class_videos and class_name != "Normal_Videos_":
                    class_scores = np.concatenate([v["scores"] for v in class_videos])
                    class_labels = np.concatenate([v["labels"] for v in class_videos])
                    try:
                        class_roc_auc = roc_auc_score(class_labels, class_scores)
                        class_precision, class_recall, _ = precision_recall_curve(class_labels, class_scores)
                        class_pr_auc = auc(class_recall, class_precision)
                        print(f"{class_name}: AUC={class_roc_auc:.4f}, AP={class_pr_auc:.4f}, Videos={len(class_videos)}")
                        save_metric(output_dir, f"roc_auc_{class_name}", num_neighbors, class_roc_auc)
                        save_metric(output_dir, f"pr_auc_{class_name}", num_neighbors, class_pr_auc)
                    except ValueError as e:
                        print(f"Skipping {class_name}: {e}")

        # Optional: Visualize class-wise AUC
    import matplotlib.pyplot as plt
    class_names = [cn for cn in class_results.keys() if cn != "Normal_Videos_" and class_results[cn]]
    auc_values = []
    for cn in class_names:
        class_scores = np.concatenate([v["scores"] for v in class_results[cn]])
        class_labels = np.concatenate([v["labels"] for v in class_results[cn]])
        try:
            auc_values.append(roc_auc_score(class_labels, class_scores))
        except ValueError:
            auc_values.append(0.0)
    plt.bar(class_names, auc_values, color='skyblue')
    plt.xlabel('Anomaly Class')
    plt.ylabel('AUC')
    plt.title('Class-Wise AUC')
    plt.ylim(0, 1)
    plt.savefig(output_dir / "class_wise_auc.png")
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--temporal_annotation_file", type=str)
    parser.add_argument("--scores_dir", type=str, required=True)
    parser.add_argument("--similarity_dir", type=str, required=True)
    parser.add_argument("--captions_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Optional arguments with defaults
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--normal_label", type=int)
    parser.add_argument("--num_neighbors", type=int, default=10)

    parser.add_argument("--without_labels", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--video_fps", type=float)

    args = parser.parse_args()
    if args.temporal_annotation_file is None and not args.without_labels:
        parser.error("--temporal_annotation_file is required when --without_labels is not used")
    if args.visualize:
        if args.video_fps is None:
            parser.error("--video_fps is required when --visualize is used")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.root_path,
        args.annotationfile_path,
        args.temporal_annotation_file,
        args.scores_dir,
        args.similarity_dir,
        args.captions_dir,
        args.output_dir,
        args.frame_interval,
        args.normal_label,
        args.num_neighbors,
        args.without_labels,
        args.visualize,
        args.video_fps,
    )

# scores_dir="${ucf_crime_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"
# echo $scores_dir
# similarity_dir="${ucf_crime_dir}/similarity/clean_summary/${llm_model_name}/${index_name}/"
# echo $similarity_dir
# output_dir="${ucf_crime_dir}/scores/refined/${llm_model_name}/${index_name}/Ubaido/"
# echo $output_dir

# python -m src.eval \
#     --root_path "$root_path" \
#     --annotationfile_path "$annotationfile_path" \
#     --temporal_annotation_file "$temporal_annotation_file" \
#     --scores_dir "$scores_dir" \
#     --similarity_dir "$similarity_dir" \
#     --captions_dir "$captions_dir" \
#     --output_dir "$output_dir" \
#     --frame_interval "$frame_interval" \
#     --normal_label "$normal_label" \
#     --num_neighbors "$num_neighbors" \
#     --video_fps "$video_fps"
