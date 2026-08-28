import argparse
import json
import sys
from pathlib import Path
import faiss
import numpy as np
import torch
from tqdm import tqdm
sys.path.append("libs/ImageBind")
from libs.ImageBind.imagebind import data
from libs.ImageBind.imagebind.models.imagebind_model import ModalityType
from src.data.video_record import VideoRecord
from src.utils.path_utils import find_unprocessed_videos
from src.utils.sample_utils import uniform_temporal_subsample
from src.utils.torch_utils import initialize_vlm_model_and_device
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class FastEnhancedVideoTextScoreRefiner:
    def __init__(
        self,
        model,
        device,
        video_embeddings_dir,
        output_scores_dir,
        output_summary_dir,
        output_similarity_dir,
        output_filenames_dir,
        num_samples,
        num_neighbors,
        index_dir,
        captions_dir,
        scores_dir,
        clip_duration,
        fps,
        imagefile_template,
        batch_size,
        frame_interval,
        visualize=True,
        initial_k=30,
        text_weight=0.5,
        visual_weight=0.5,
        temporal_window=5
    ):
        self.model = model
        self.device = device
        self.video_embeddings_dir = Path(video_embeddings_dir)
        self.output_scores_dir = Path(output_scores_dir)
        self.output_summary_dir = Path(output_summary_dir)
        self.output_similarity_dir = Path(output_similarity_dir)
        self.output_filenames_dir = Path(output_filenames_dir)
        self.num_samples = num_samples
        self.num_neighbors = num_neighbors

        self.initial_k = max(initial_k or max(30, num_neighbors * 3), num_neighbors)
        self.text_weight = text_weight
        self.visual_weight = visual_weight
        self.temporal_window = temporal_window
        self.index_dir = index_dir
        self.captions_dir = Path(captions_dir)
        self.scores_dir = Path(scores_dir)
        self.clip_duration = clip_duration
        self.fps = fps
        self.imagefile_template = imagefile_template
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.visualize = visualize

        # CACHES
        self.video_embeddings_cache = {}
        self.idx2frames_cache = {}   # NEW: FAISS idx → [all frames]

        print(f"Enhanced Retrieval Config: K={self.num_neighbors}, Initial_K={self.initial_k}, "
              f"TextW={self.text_weight}, VisW={self.visual_weight}, TempWin={self.temporal_window}")

    # ====================== LOADERS ======================
    def load_video_embeddings(self, video):
        video_name = Path(video.path).name
        if video_name in self.video_embeddings_cache:
            return self.video_embeddings_cache[video_name]
        file = self.video_embeddings_dir / f"{video_name}.npz"
        if not file.exists():
            raise FileNotFoundError(f"Embeddings not found: {file}")
        data = np.load(file)
        emb_data = {
            "embeddings": data["embeddings"],
            "frame_indices": data["frame_indices"].tolist(),
            "norms": data["norms"]
        }
        self.video_embeddings_cache[video_name] = emb_data
        return emb_data

    def _load_faiss_index(self, video_name):
        path = Path(self.index_dir) / f"{video_name}.bin"
        return faiss.read_index(str(path))

    def _load_file_names(self, video_name):
        path = Path(self.index_dir) / f"{video_name}.json"
        with open(path) as f:
            return json.load(f)

    def _load_idx2frames(self, video_name):
        """Load FAISS index → all frame indices that share the caption"""
        if video_name in self.idx2frames_cache:
            return self.idx2frames_cache[video_name]
        path = Path(self.index_dir) / f"{video_name}_idx2frames.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run updated index builder!")
        with open(path) as f:
            data = json.load(f)
            # Convert keys to int
            self.idx2frames_cache[video_name] = {int(k): v for k, v in data.items()}
        return self.idx2frames_cache[video_name]

    def load_text_embeddings(self, video_name):
        index = self._load_faiss_index(video_name)
        return index.reconstruct_n(0, index.ntotal)

    # ====================== VISUAL SIM ======================
    def _compute_visual_similarity(self, query_emb, video_emb_data, query_frame_idx):
        video_embs = video_emb_data['embeddings']
        frame_idxs = video_emb_data['frame_indices']
        sims = cosine_similarity(query_emb.reshape(1, -1), video_embs).flatten()

        weights = np.ones_like(sims)
        for i, fidx in enumerate(frame_idxs):
            dist = abs(fidx - query_frame_idx)
            if dist <= self.temporal_window * self.frame_interval:
                weights[i] = 1.0 + 0.5 * np.exp(-dist / (self.frame_interval * 2))
        return sims * weights, frame_idxs

    # ====================== RE-RANKING ======================
    def _cross_modal_reranking(self, 
                              query_emb, query_frame_idx,
                              text_dists, text_idxs,
                              video_emb_data, text_embs, file_names, idx2frames):
        candidates = text_idxs[:self.initial_k]
        text_dists = text_dists[:self.initial_k]

        vis_sims, frame_idxs = self._compute_visual_similarity(query_emb, video_emb_data, query_frame_idx)
        frame_to_vis = dict(zip(frame_idxs, vis_sims))

        scores = []
        for i, cand_idx in enumerate(candidates):
            text_sim = 1.0 / (1.0 + text_dists[i])

            # GET ALL FRAMES FOR THIS CAPTION
            all_frames = idx2frames.get(cand_idx, [])
            if not all_frames:
                rep_file = file_names[cand_idx]
                all_frames = [int(rep_file.split("/")[-1])]

            # MAX VISUAL SIM (or use weighted avg below)
            visual_sim = max(frame_to_vis.get(f, 0.0) for f in all_frames)


            combined = self.text_weight * text_sim + self.visual_weight * visual_sim
            scores.append({
                'candidate_idx': cand_idx,
                'frame_idx': all_frames[0],
                'all_frames': all_frames,
                'text_sim': text_sim,
                'visual_sim': visual_sim,
                'combined_score': combined,
                'text_distance': text_dists[i]
            })

        scores.sort(key=lambda x: x['combined_score'], reverse=True)
        final = scores[:self.num_neighbors]
        final_idxs = [c['candidate_idx'] for c in final]
        final_dists = [c['text_distance'] for c in final]
        return np.array([final_dists]), np.array([final_idxs]), final

    # ====================== BATCH RETRIEVAL ======================
    def _retrieve_captions_enhanced(self,
                                  query_embs, batch_frame_idxs,
                                  file_names, video_captions,
                                  video_captions_nn, video_similarity_nn, ret_file_names_nn,
                                  index, video_emb_data, text_embs, idx2frames):
        for q_emb, q_frame in zip(query_embs, batch_frame_idxs):
            faiss.normalize_L2(q_emb.reshape(1, -1))
            D, I = index.search(q_emb.reshape(1, -1), self.initial_k)

            final_D, final_I, details = self._cross_modal_reranking(
                q_emb, q_frame, D[0], I[0],
                video_emb_data, text_embs, file_names, idx2frames
            )

            caps, sims, names = {}, {}, {}
            for i in range(self.num_neighbors):
                file_name = file_names[final_I[0][i]]
                names[str(i)] = file_name
                sims[str(i)] = final_D[0][i].item()
                ret_idx = file_name.split("/")[-1]
                caps[str(i)] = video_captions[ret_idx]

            ret_file_names_nn[str(q_frame)] = names
            video_captions_nn[str(q_frame)] = caps
            video_similarity_nn[str(q_frame)] = sims

    # ====================== MAIN ======================
    def retrieve_nn(self, video: VideoRecord):
        video_name = Path(video.path).name
        video_captions_nn = {}
        video_similarity_nn = {}
        ret_file_names_nn = {}
        video_magnitudes = {}

        # Load data
        with open(self.captions_dir / f"{video_name}.json") as f:
            video_captions = json.load(f)

        index = self._load_faiss_index(video_name)
        file_names = self._load_file_names(video_name)
        text_embs = self.load_text_embeddings(video_name)
        idx2frames = self._load_idx2frames(video_name)  # NEW

        video_emb_data = self.load_video_embeddings(video)
        frame_to_emb = dict(zip(video_emb_data["frame_indices"], video_emb_data["embeddings"]))
        frame_to_norm = dict(zip(video_emb_data["frame_indices"], video_emb_data["norms"]))

        for start in tqdm(range(0, video.num_frames, self.batch_size * self.frame_interval),
                          desc=f"Processing {video.path}", unit="batch"):
            end = min(start + self.batch_size * self.frame_interval, video.num_frames)
            batch_idxs = list(range(start, end, self.frame_interval))
            query_embs = np.array([frame_to_emb[f] for f in batch_idxs])
            for f in batch_idxs:
                video_magnitudes[str(f)] = float(frame_to_norm[f])

            self._retrieve_captions_enhanced(
                query_embs, batch_idxs,
                file_names, video_captions,
                video_captions_nn, video_similarity_nn, ret_file_names_nn,
                index, video_emb_data, text_embs, idx2frames
            )

        self._save_results(video_name, video_captions_nn, video_similarity_nn, ret_file_names_nn, video_magnitudes)
        if self.visualize:
            self.visualize_embeddings_enhanced(video, video_emb_data, text_embs)

    def visualize_embeddings_enhanced(self, video, video_embeddings_data, text_embeddings):
        """Enhanced visualization showing the improved retrieval quality."""
        video_name = Path(video.path).name
        video_embeddings = video_embeddings_data['embeddings']
        
        # Stack embeddings for joint visualization
        all_embeddings = np.vstack([video_embeddings, text_embeddings])
        
        # Create labels (0 for video, 1 for text)
        labels = np.array([0] * len(video_embeddings) + [1] * len(text_embeddings))
        
        # Reduce to 2D using t-SNE
        print("Applying t-SNE for enhanced visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # Create subplots for comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        colors = ['skyblue', 'darkorange']
        label_names = ['Video Clip Embeddings', 'Text Embeddings']
        
        # Plot 1: Original embedding space
        for i in [0, 1]:
            mask = labels == i
            ax1.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                s=20 if i == 0 else 5,
                alpha=0.7,
                label=label_names[i],
                color=colors[i]
            )
        
        ax1.set_xlabel("t-SNE Dimension 1")
        ax1.set_ylabel("t-SNE Dimension 2")
        ax1.set_title(f"Enhanced Video-Text Embedding Space\n{video_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        
        
        # Save plot
        output_plot_path = self.output_summary_dir / f"{video_name}_enhanced_retrieval_analysis.png"
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced visualization saved to: {output_plot_path}")
        plt.show()

    def _save_results(self, video_name, video_captions_nn, video_similarity_nn, ret_file_names_nn,video_magnitudes):
        """Save results with enhanced metadata."""
        # Save original results
        output_path = self.output_summary_dir / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_captions_nn, f, indent=4)

        output_path = self.output_similarity_dir / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_similarity_nn, f, indent=4)

        output_path = self.output_filenames_dir / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(ret_file_names_nn, f, indent=4)
        output_path=self.output_summary_dir/f"{video_name}_magnitudes.json"
        output_path.parent.mkdir(parents=True,exist_ok=True)
        with open(output_path,"w") as f:
            json.dump(video_magnitudes,f,indent=4)
        # Save retrieval configuration
        config = {
            'text_weight': self.text_weight,
            'visual_weight': self.visual_weight,
            'initial_k': self.initial_k,
            'temporal_window': self.temporal_window,
            'num_neighbors': self.num_neighbors
        }
        config_path = self.output_summary_dir / f"{video_name}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    # Keep all other methods from original class...
    def _load_scores(self, video_name):
        scores_file_path = self.scores_dir / f"{video_name}.json"
        with open(scores_file_path) as f:
            return json.load(f)

    def _load_ret_file_names_nn(self, video_name):
        ret_file_names_nn_file_path = self.output_filenames_dir / f"{video_name}.json"
        with open(ret_file_names_nn_file_path) as f:
            return json.load(f)

    def _save_scores(self, video_name, video_scores_nn):
        output_path = self.output_scores_dir / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_scores_nn, f, indent=4)

    def refine_scores(self, video: VideoRecord):
        video_name = Path(video.path).name
        video_scores_nn = {}

        video_scores = self._load_scores(video_name)
        ret_file_names_nn = self._load_ret_file_names_nn(video_name)

        for batch_start_frame in tqdm(
            range(0, video.num_frames, self.batch_size * self.frame_interval),
            desc=f"Processing {video.path}",
            unit="batch",
        ):
            batch_end_frame = min(
                batch_start_frame + (self.batch_size * self.frame_interval), video.num_frames
            )
            batch_center_frame_idxs = range(
                batch_start_frame, batch_end_frame, self.frame_interval
            )

            for idx, frame_idx in enumerate(batch_center_frame_idxs):
                frame_scores = {}
                for nn_idx in range(self.num_neighbors):
                   # import ipdb;ipdb.set_trace()
                    file_name = ret_file_names_nn[str(frame_idx)][str(nn_idx)]
                    
                    ret_index = file_name.split("/")[-1]
                    frame_scores[str(nn_idx)] = video_scores[ret_index]
                video_scores_nn[str(frame_idx)] = frame_scores

        self._save_scores(video_name, video_scores_nn)


def run_enhanced(
    root_path,
    annotationfile_path,
    batch_size,
    frame_interval,
    video_embeddings_dir,
    output_scores_dir,
    output_summary_dir,
    output_similarity_dir,
    output_filenames_dir,
    captions_dir,
    index_dir,
    scores_dir,
    resume,
    pathname,
    imagefile_template,
    fps,
    clip_duration,
    num_samples,
    num_neighbors,
    num_jobs,
    job_id,
    visualize=True,
    # Enhanced parameters
    initial_k=None,
    text_weight=0.5,
    visual_weight=0.5,
    temporal_window=5
):
    model, device = initialize_vlm_model_and_device()

    enhanced_refiner = FastEnhancedVideoTextScoreRefiner(
        model,
        device,
        video_embeddings_dir,
        output_scores_dir,
        output_summary_dir,
        output_similarity_dir,
        output_filenames_dir,
        num_samples,
        num_neighbors,
        index_dir,
        captions_dir,
        scores_dir,
        clip_duration,
        fps,
        imagefile_template,
        batch_size,
        frame_interval,
        visualize=visualize,
        initial_k=initial_k,
        text_weight=text_weight,
        visual_weight=visual_weight,
        temporal_window=temporal_window
    )

    video_list = [VideoRecord(x.strip().split(), root_path) for x in open(annotationfile_path)]
    video_list = np.array_split(video_list, num_jobs)[job_id]
    if resume:
        video_list = find_unprocessed_videos(video_list, output_summary_dir, pathname)

    for video in video_list:
        enhanced_refiner.retrieve_nn(video)
        enhanced_refiner.refine_scores(video)


def parse_args_enhanced():
    parser = argparse.ArgumentParser()
    # Original arguments
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--video_embeddings_dir",type=str,required=True)
    parser.add_argument("--output_scores_dir", type=str, required=True)
    parser.add_argument("--output_summary_dir", type=str, required=True)
    parser.add_argument("--output_similarity_dir", type=str, required=True)
    parser.add_argument("--output_filenames_dir", type=str, required=True)
    parser.add_argument("--captions_dir", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--scores_dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pathname", type=str, default="*.json")
    parser.add_argument("--imagefile_template", type=str, default="{:06d}.jpg")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--clip_duration", type=float, default=10)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_neighbors", type=int, default=1)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--visualize", action="store_true", help="Generate t-SNE visualization")
    
    # Enhanced retrieval parameters
    parser.add_argument("--initial_k", type=int, default=None, 
                       help="Initial number of candidates to retrieve from text search (default: 5x num_neighbors)")
    parser.add_argument("--text_weight", type=float, default=0.5,
                       help="Weight for text similarity in combined score")
    parser.add_argument("--visual_weight", type=float, default=0.5,
                       help="Weight for visual similarity in combined score")
    parser.add_argument("--temporal_window", type=int, default=5,
                       help="Temporal window for boosting nearby frame similarities")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args_enhanced()
    run_enhanced(
        args.root_path,
        args.annotationfile_path,
        args.batch_size,
        args.frame_interval,
        args.video_embeddings_dir,
        args.output_scores_dir,
        args.output_summary_dir,
        args.output_similarity_dir,
        args.output_filenames_dir,
        args.captions_dir,
        args.index_dir,
        args.scores_dir,
        args.resume,
        args.pathname,
        args.imagefile_template,
        args.fps,
        args.clip_duration,
        args.num_samples,
        args.num_neighbors,
        args.num_jobs,
        args.job_index,
        visualize=args.visualize,
        initial_k=args.initial_k,
        text_weight=args.text_weight,
        visual_weight=args.visual_weight,
        temporal_window=args.temporal_window
    )
