import os
import json
import cv2
import logging
import numpy as np
import subprocess
from tqdm import tqdm
import torch
import re  
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import List, Tuple, Any, Dict, Optional
import ffmpeg
from pathlib import Path
import argparse
from dataclasses import dataclass
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed environment variable name
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PATH"] += os.pathsep + "/mnt/fast/nobackup/users/mw01832/ffmpeg/"

class VideoRecord:
    """Represents a video record with frame information."""
    
    def __init__(self, row: List[str], root_path: str):
        self.video_name = row[0]  # e.g., Abuse028_x264
        self.start_frame = int(row[1])  # e.g., 0
        self.end_frame = int(row[2])  # e.g., 1411
        self.label = int(row[3])  # e.g., 0
        self.root_path = os.path.normpath(root_path)
        self.path = os.path.normpath(os.path.join(self.root_path, self.video_name))
        
        # Initialize frame count
        self._validate_and_count_frames()
    
    def _validate_and_count_frames(self) -> None:
        """Validate frame directory and count available frames."""
        frame_dir = Path(self.path)
        logger.debug(f"Checking frame directory: {frame_dir}")
        
        if not frame_dir.exists():
            logger.error(f"Frame directory {frame_dir} does not exist for {self.video_name}")
            self.num_frames = 0
            return
        
        frame_files = sorted(frame_dir.glob("*.jpg"))
        self.num_frames = len(frame_files)
        expected_num_frames = self.end_frame - self.start_frame + 1
        
        if self.num_frames != expected_num_frames:
            logger.warning(
                f"Frame count mismatch for {self.video_name}: "
                f"Expected {expected_num_frames}, found {self.num_frames}"
            )
        
        if self.num_frames == 0:
            logger.warning(f"No frames found in {frame_dir} for {self.video_name}")


def uniform_temporal_subsample(clip_frame_paths: List[Any], num_samples: int) -> List[Any]:
    """Uniformly subsample frames from a clip."""
    t = len(clip_frame_paths)
    assert num_samples > 0 and t > 0, f"Invalid parameters: num_samples={num_samples}, t={t}"

    if num_samples >= t:
        return clip_frame_paths

    # Calculate indices for subsampling
    indices = np.linspace(0, t - 1, num_samples, dtype=int)
    indices = np.clip(indices, 0, t - 1)

    # Select frames using calculated indices
    subsampled_frames = [clip_frame_paths[i] for i in indices]
    return subsampled_frames
    

def find_unprocessed_videos(video_list: List[VideoRecord], output_dir: str, pattern: str) -> List[VideoRecord]:
    """Find videos that haven't been processed yet."""
    output_path = Path(output_dir)
    processed_files = set(f.stem for f in output_path.glob(pattern))
    
    unprocessed = [
        video for video in video_list 
        if video.video_name not in processed_files
    ]
    
    logger.info(f"Found {len(unprocessed)} unprocessed videos out of {len(video_list)}")
    return unprocessed
    

@dataclass
class AnomalyAnalysis:
    description: str
    reasoning: str
    explanation: str
    anomaly_score: float
    anomaly_type: str
    raw_response: str

class ChainOfThoughtAnomalyProcessor:
    """Handles chain-of-thought reasoning for video anomaly detection"""
    
    def __init__(self, anomaly_db=None):
        self.anomaly_db = anomaly_db or self._get_default_anomaly_db()
        
    def _get_default_anomaly_db(self) -> dict:
        return {
            "NORMAL": "Regular, expected activities with no safety concerns like normal waliking ,playing ,shopping,talking, cooking, etc",
            "VIOLENCE": "Physical aggression, hitting, kicking, fighting between people", 
            "ASSAULT": "Physical attack on a person with violent contact",
            "FIGHTING": "Physical confrontation between multiple people",
            "ROBBERY": "Theft with force or threat, forcefully grabbing items or valuables",
            "SHOOTING": "Use of firearms, pointing weapons at people",
            "VANDALISM": "Intentionally damaging property, graffiti, breaking objects",
            "SHOPLIFTING": "Stealing items from stores with concealing movements",
            "BURGLARY": "Breaking into or entering buildings illegally",
            "ARSON": "Deliberately setting fires to property",
            "EXPLOSION": "Blast, fire, smoke indicating explosive event",
            "ARREST": "Law enforcement restraining or detaining individuals",
            "ACCIDENT": "Unintentional harmful incidents like falls, collisions, crashes",
            "SUSPICIOUS": "Unusual behavior that raises concern but doesn't clearly fit other categories"
        }

    def build_cot_prompt(self, prev_context: Optional[Dict] = None) -> str:
        """Build chain-of-thought prompt with optional temporal context"""
        anomaly_types = list(self.anomaly_db.keys())
        anomaly_descriptions = '\n'.join([f"  • {k}: {v}" for k, v in self.anomaly_db.items()])

        prompt = f"""You are an expert video surveillance analyst identifying unusual events and potential safety issues and anomalies and to rate the video clip in the range [0.000 to 1.000] as an anomlay score based on your analysis. Let's think step by step to analyze this video clip.
      
    ANOMALY CATEGORIES:
    {anomaly_descriptions}"""

        if prev_context and prev_context.get('prev_description'):
            prompt += f"""

                PREVIOUS CLIP CONTEXT:
                - Previous scene: {prev_context['prev_description']}
                - Previous type: {prev_context.get('prev_type', 'NORMAL')}
                - Previous score: {prev_context.get('prev_score', 0.0):.3f}
                
        Consider temporal consistency. If the current clip shows similar activities, maintain similar ANOMALY_TYPE and score, but rephrase the DESCRIPTION to reflect only what you observe in this clip."""    
        prompt += """

            Let's analyze this video step-by-step. Provide your response in this EXACT format:

            DESCRIPTION: [Write 2 clear sentences describing ONLY what you observe: people, objects, actions, environment. Be factual and specific. Focus on what is happening, not what is NOT happening.]

            REASONING: [Think through this step-by-step:
            1) What specific actions are taking place?
            2) Are these actions normal for this environment or unusual/threatening?
            3) Do the actions match any anomaly category? Which one and why?
            4) What is the severity level - minor, moderate, or severe?]

            EXPLANATION: [Based on your reasoning above, provide a clear 1-2 sentence explanation of why this scene is normal or anomalous. Be direct and specific.]

            ANOMALY_SCORE: [Single number between 0.000 and 1.000 low score means normal and high score means abnormal or unusual or potential safety issues]
            ANOMALY_TYPE: [See what best suits for this video clip. Choose ONE if available otherwise provide yourself: {', '.join(anomaly_types)}]
           

            SCORING GUIDELINES:
            • 0.000 = Completely normal, everyday activities with no concerns whatsoever
            • 1.000 = Severe criminal activity or emergency requiring immediate intervention

            Your score should reflect the SEVERITY and THREAT LEVEL of what you observe on a continuous scale between these two extremes. Consider:
            - How dangerous or criminal is the activity?
            - What is the potential for harm to people or property?
            - How urgent is the need for intervention?

            Assign scores based on your step-by-step reasoning. The score should naturally emerge from your analysis of the actions, context, and severity. There are no fixed boundaries - use the full continuous range from 0.000 to 1.000 based on your judgment.

            CRITICAL INSTRUCTIONS:
            - Think step-by-step through your reasoning
            - Be specific about what you observe
            - Base your score on the severity of the activity
            - Use the EXACT format shown above
            - Provide all fields (DESCRIPTION, REASONING, EXPLANATION, ANOMALY_SCORE, ANOMALY_TYPE)
            Let's begin the analysis:"""

        return prompt
    

class VideoAnomalyDetector:
    """Generate raw anomaly detection scores using VideoLLaMA3 with temporal consistency."""
    
    def __init__(
        self,
        model,
        processor,
        device: str,
        root_path: str,
        output_scores_dir: str,
        batch_size: int = 1,
        frame_interval: int = 16,
        fps: float = 30.0,  # Original FPS from arguments
        fps_create: float = 1.0,  # FPS for creating template video clips
        fps_vlama: float = 1.0,  # FPS for VideoLLaMA3 processing
        clip_duration: float = 10.0,
        imagefile_template: str = "{:06d}.jpg",
        max_new_tokens: int = 150,
        temporal_consistency_weight: float = 0.3 , # Weight for temporal smoothing
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.root_path = Path(root_path)
        self.output_scores_dir = Path(output_scores_dir)
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.fps = fps  # Original FPS (e.g., 30)
        self.fps_create = fps_create  # FPS for template video creation (1)
        self.fps_vlama = fps_vlama  # FPS for VideoLLaMA3 (1)
        self.clip_duration = clip_duration
        self.imagefile_template = imagefile_template
        self.max_new_tokens = max_new_tokens
        self.temporal_consistency_weight = temporal_consistency_weight
        self.cot_processor=ChainOfThoughtAnomalyProcessor()
        
        # Reset temporal tracking
        self._last_analysis = None
        
        # Create temp directory for videos
        self.temp_dir = Path("./temp_videos")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Ensure output directory exists
        self.output_scores_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_available_frame_paths(self, frame_dir: Path) -> List[Path]:
        """Get list of available frame file paths sorted by frame number."""
        if not frame_dir.exists():
            logger.error(f"Frame directory {frame_dir} does not exist")
            return []

        frame_files = sorted(frame_dir.glob("*.jpg"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
        
        if not frame_files:
            logger.error(f"No valid frame files found in {frame_dir}")
        
        return frame_files
         
    def _get_clip_frame_paths_around_center(
        self, 
        video: VideoRecord, 
        center_frame_idx: int, 
        frames_per_clip: int = 150
    ) -> List[Path]:
        """Get frame paths for a clip centered around the given frame index."""
        frame_dir = Path(os.path.normpath(video.path))
        available_frame_paths = self._get_available_frame_paths(frame_dir)
        
        if not available_frame_paths:
            return []

        logger.debug(f"Found {len(available_frame_paths)} frame paths in {frame_dir}")

        # Calculate the window around center frame
        start_frame = max(center_frame_idx - frames_per_clip // 2, 0)
        end_frame = min(center_frame_idx + frames_per_clip // 2, len(available_frame_paths))
        
        # Get frame paths in the window
        window_frame_paths = available_frame_paths[start_frame:end_frame]

        if not window_frame_paths:
            logger.warning(f"No valid frame paths for center_frame={center_frame_idx}")
            return []
        
        # Subsample to exactly 10 frames
        try:
            clip_frame_paths = uniform_temporal_subsample(window_frame_paths, 10)
        except AssertionError as e:
            logger.error(f"Failed to subsample frames for center_frame={center_frame_idx}: {e}")
            clip_frame_paths = window_frame_paths[:10] if len(window_frame_paths) >= 10 else window_frame_paths

        # Pad if fewer than 10 frames
        while len(clip_frame_paths) < 10 and clip_frame_paths:
            clip_frame_paths.append(clip_frame_paths[-1])

        logger.debug(f"Center frame {center_frame_idx}: selected {len(clip_frame_paths)} frame paths")
        return clip_frame_paths[:10]
        
    def _load_frames_from_paths(self, frame_paths: List[Path]) -> List[np.ndarray]:
        """Load frames directly from frame paths."""
        try:
            frames = []
            for frame_path in frame_paths:
                if not frame_path.exists():
                    logger.warning(f"Frame file does not exist: {frame_path}")
                    continue
                    
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    # Convert BGR to RGB for consistency
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb.astype(np.uint8))
                else:
                    logger.warning(f"Could not load frame: {frame_path}")

            # Pad with last frame if needed
            while len(frames) < 10 and frames:
                frames.append(frames[-1])

            return frames
        except Exception as e:
            logger.error(f"Error loading frames from paths: {e}")
            return []
    # ... existing code ...
  
        
    def _create_temp_video_from_frame_paths(
        self, 
        frame_paths: List[Path], 
        center_frame_idx: int
    ) -> Optional[str]:
        """Create temporary video file from frame paths using fps_create."""
        if not frame_paths:
            return None
            
         # Determine output path
  
        temp_video_path = self.temp_dir / f"temp_clip_{center_frame_idx}_{os.getpid()}.mp4"
        
        try:
            # Load frames from paths
            frames = self._load_frames_from_paths(frame_paths)
            if not frames:
                logger.error(f"No frames loaded from paths for center frame {center_frame_idx}")
                return None
                
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Use fps_create (1.0) for template video creation
            out = cv2.VideoWriter(str(temp_video_path), fourcc, self.fps_create, (width, height))
            
            for frame in frames:
                # Convert RGB back to BGR for cv2.VideoWriter
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            
            out.release()    
            
            # Verify the video was created successfully
            if not temp_video_path.exists() or temp_video_path.stat().st_size == 0:
                logger.error(f"Failed to create valid temp video: {temp_video_path}")
                return None 
                
            logger.debug(f"Created temp video: {temp_video_path} with fps_create={self.fps_create}")
            return str(temp_video_path)
            
        except Exception as e:
            logger.error(f"Error creating temp video from frame paths: {e}")
            if temp_video_path.exists():
                temp_video_path.unlink()
            return None

    def _create_cot_anomaly_prompt(
        self,
        center_frame_idx: int,
        prev_analysis: Optional[AnomalyAnalysis] = None
    ) -> str:
        """Create COT anomaly detection prompt with temporal context."""
        prev_context = None
        if prev_analysis:
            prev_context = {
                'prev_description': prev_analysis.description,
                'prev_type': prev_analysis.anomaly_type,
                'prev_score': prev_analysis.anomaly_score
            }
        
        return self.cot_processor.build_cot_prompt(prev_context=prev_context)
        
    
                   
    def _create_videollama_conversation(
        self, 
        temp_video_path: str, 
        center_frame_idx: int, 
        num_frames: int,
        prev_analysis: Optional[AnomalyAnalysis] = None
    ) -> List[Dict[str, Any]]:
        """Create conversation format for VideoLLaMA3 using fps_vlama."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant specialized in video analysis and anomaly detection."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": temp_video_path,
                            "fps": self.fps_vlama,  
                            "max_frames": num_frames
                        }
                    },
                    {
                        "type": "text",
                        "text": self._create_cot_anomaly_prompt(
                            center_frame_idx,  
                            prev_analysis=prev_analysis  #
                        )
                    }
                ]
            }
        ]
        return conversation
    
    def _cleanup_temp_video(self, temp_video_path: str) -> None:
        """Clean up temporary video file."""
       
        try:
            path = Path(temp_video_path)
            if path.exists():
                path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete temp video {temp_video_path}: {e}")
                       
    def _parse_videollama_output(self, output: str, default_score: float = 0.0) -> AnomalyAnalysis:
        """Parse VideoLLaMA3 COT output with refusal handling."""
        try:
            # Check for refusals FIRST
            refusal_phrases = ["can't assist", "cannot assist", "i'm sorry", "i cannot", "i apologize"]
            if any(phrase in output.lower() for phrase in refusal_phrases):
                logger.warning("Model refused - reusing previous analysis")
                
                if self._last_analysis:
                    # Return previous analysis with note about refusal
                    return AnomalyAnalysis(
                        description=self._last_analysis.description,
                        reasoning=f"Model refused. Previous context maintained.",
                        explanation=self._last_analysis.explanation,
                        anomaly_score=self._last_analysis.anomaly_score,
                        anomaly_type=self._last_analysis.anomaly_type,
                        raw_response=f"REFUSAL: {output}"
                    )
                else:
                    # No previous analysis available
                    return AnomalyAnalysis(
                        description="Model declined to analyze",
                        reasoning="Safety filter triggered",
                        explanation="No previous context available",
                        anomaly_score=0.0,
                        anomaly_type="NORMAL",
                        raw_response=output
                    )
            
            # Normal parsing
            lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
            
            description = "No description provided"
            reasoning = "No reasoning provided"
            explanation = "No explanation provided"
            anomaly_score = default_score
            anomaly_type = "NORMAL"
            raw_response = output
            
            current_section = None
            for line in lines:
                line_lower = line.lower()
                
                # Detect section headers
                if line_lower.startswith('description:'):
                    current_section = 'description'
                    description = line.split(':', 1)[1].strip() if ':' in line else line
                elif line_lower.startswith('reasoning:'):
                    current_section = 'reasoning'
                    reasoning = line.split(':', 1)[1].strip() if ':' in line else line
                elif line_lower.startswith('explanation:'):
                    current_section = 'explanation'
                    explanation = line.split(':', 1)[1].strip() if ':' in line else line
                elif line_lower.startswith('anomaly_score:'):
                    score_text = line.split(':', 1)[1].strip()
                    score_match = re.search(r'(\d*\.?\d+)', score_text)
                    if score_match:
                        anomaly_score = round(max(0.0, min(1.0, float(score_match.group(1)))), 3)
                    current_section = None
                elif line_lower.startswith('anomaly_type:'):
                    anomaly_type = line.split(':', 1)[1].strip().upper()
                    if anomaly_type not in self.cot_processor.anomaly_db:
                        anomaly_type = "NORMAL"
                    current_section = None
                
                # Accumulate multi-line sections
                elif current_section:
                    if current_section == 'description':
                        description += " " + line
                    elif current_section == 'reasoning':
                        reasoning += " " + line
                    elif current_section == 'explanation':
                        explanation += " " + line
            
            # Apply temporal smoothing
            if self._last_analysis is not None:
                prev_score = self._last_analysis.anomaly_score
                smoothed_score = (
                    self.temporal_consistency_weight * prev_score + 
                    (1 - self.temporal_consistency_weight) * anomaly_score
                )
                anomaly_score = round(smoothed_score, 3)
            
            analysis = AnomalyAnalysis(
                description=description.strip(),
                reasoning=reasoning.strip(),
                explanation=explanation.strip(),
                anomaly_score=anomaly_score,
                anomaly_type=anomaly_type,
                raw_response=raw_response
            )
            
            # Update for next iteration
            self._last_analysis = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing COT output: {e}")
            
            # On parsing error, use previous if available
            if self._last_analysis:
                return AnomalyAnalysis(
                    description=self._last_analysis.description,
                    reasoning=f"Parsing error: {str(e)}. Using previous.",
                    explanation=self._last_analysis.explanation,
                    anomaly_score=self._last_analysis.anomaly_score,
                    anomaly_type=self._last_analysis.anomaly_type,
                    raw_response=output
                )
            else:
                return AnomalyAnalysis(
                    description="Parsing error",
                    reasoning=f"Error: {str(e)}",
                    explanation="Failed to analyze clip",
                    anomaly_score=round(default_score, 3),
                    anomaly_type="NORMAL",
                    raw_response=output
                )  
    
    def _validate_video_frames(self, video: VideoRecord) -> bool:
        """Validate video frame information."""
        frame_dir = Path(os.path.normpath(video.path))
        available_frame_paths = self._get_available_frame_paths(frame_dir)
        actual_num_frames = len(available_frame_paths)
        
        if actual_num_frames == 0:
            logger.error(f"No frames available for video {video.video_name}")
            return False
        
        expected_num_frames = video.end_frame - video.start_frame + 1
        if actual_num_frames != video.num_frames or actual_num_frames != expected_num_frames:
            logger.warning(
                f"Frame count mismatch for {video.video_name}: "
                f"Expected {expected_num_frames}, VideoRecord reports {video.num_frames}, "
                f"found {actual_num_frames} frames"
            )
            video.num_frames = actual_num_frames
        
        return True

    def generate_raw_anomaly_scores(self, video: VideoRecord) -> None:
        """Generate raw anomaly scores for a video with temporal consistency."""
        video_name = video.video_name

        logger.info(f"Processing video: {video_name}")
        logger.debug(
            f"Video details - frames: {video.num_frames}, "
            f"start: {video.start_frame}, end: {video.end_frame}, label: {video.label}"
        )

        if not self._validate_video_frames(video):
            logger.error(f"Skipping {video_name}: Frame validation failed")
            return

        frames_per_clip = int(self.clip_duration * self.fps)  
        logger.debug(f"frames_per_clip calculated as: {frames_per_clip} (clip_duration={self.clip_duration} * fps={self.fps})")

        raw_scores = {}
        # Reset temporal tracking for each video (COT version)
        self._last_analysis = None
        prev_analysis = None

        logger.info(f"Processing {video_name} with {video.num_frames} frames")

        for batch_start_frame in tqdm(
            range(0, video.num_frames, self.batch_size * self.frame_interval),
            desc=f"Processing {video_name}",
            unit="batch",
        ):
            batch_end_frame = min(
                batch_start_frame + (self.batch_size * self.frame_interval), 
                video.num_frames
            )
            batch_center_frame_idxs = list(range(
                batch_start_frame, batch_end_frame, self.frame_interval
            ))

            logger.debug(f"Processing batch: frames {batch_start_frame}-{batch_end_frame}")

            for center_frame_idx in batch_center_frame_idxs:
                frame_key = str(center_frame_idx)
                
                clip_frame_paths = self._get_clip_frame_paths_around_center(
                    video, center_frame_idx, frames_per_clip
                )

                if not clip_frame_paths:
                    logger.warning(f"No clip frame paths for center frame {center_frame_idx}")
                    continue

                temp_video_path = self._create_temp_video_from_frame_paths(clip_frame_paths, center_frame_idx)
                if not temp_video_path:
                    logger.error(f"Failed to create temp video for center frame {center_frame_idx}")
                    continue

                logger.debug(f"Processing center frame {center_frame_idx}: {len(clip_frame_paths)} frame paths")

                try:
                    conversation = self._create_videollama_conversation(
                        temp_video_path, center_frame_idx, len(clip_frame_paths),
                        prev_analysis=prev_analysis  # Fixed: use COT param
                    )

                    # Process with VideoLLaMA3
                    inputs = self.processor(
                        conversation=conversation,
                        return_tensors="pt",
                        add_system_prompt=True,
                        add_generation_prompt=True
                    )

                    # Move inputs to device
                    inputs = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in inputs.items()
                    }

                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                    # Generate response
                    with torch.no_grad():
                        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

                    output = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    analysis = self._parse_videollama_output(output, prev_analysis.anomaly_score if prev_analysis else 0.0)  # Fixed: returns AnomalyAnalysis

                    raw_scores[frame_key] = {
                        "analysis": {  # Fixed: store full COT analysis
                            "description": analysis.description,
                            "reasoning": analysis.reasoning,
                            "explanation": analysis.explanation,
                            "anomaly_score": round(analysis.anomaly_score, 3),
                            "anomaly_type": analysis.anomaly_type,
                            "raw_response": analysis.raw_response
                        },
                        "prev_context": {
                            "prev_description": prev_analysis.description if prev_analysis else "None",
                            "prev_type": prev_analysis.anomaly_type if prev_analysis else "None",
                            "prev_score": round(prev_analysis.anomaly_score, 3) if prev_analysis else "None"
                        },
                        "clip_info": {
                            "center_frame": int(center_frame_idx),
                            "num_frames_in_clip": int(len(clip_frame_paths)),
                            "frame_paths": [str(p) for p in clip_frame_paths],
                            "fps_create": float(self.fps_create),
                            "fps_vlama": float(self.fps_vlama)
                        }
                    }

                    logger.info(f"✅ Frame {center_frame_idx}: Score={analysis.anomaly_score:.3f}, Type={analysis.anomaly_type}, Desc='{analysis.description[:50]}...'")  # Fixed log

                    # Update context for next iteration
                    prev_analysis = analysis

                except Exception as e:
                    logger.error(f"❌ Error processing clip centered at frame {center_frame_idx}: {e}")
                    # Fixed: fallback AnomalyAnalysis
                    fallback_analysis = AnomalyAnalysis(
                        description=f"Error: {str(e)}",
                        reasoning="Processing failed",
                        explanation="Could not analyze clip",
                        anomaly_score=round(prev_analysis.anomaly_score if prev_analysis else 0.0, 3),
                        anomaly_type=prev_analysis.anomaly_type if prev_analysis else "NORMAL",
                        raw_response=f"Error: {str(e)}"
                    )
                    raw_scores[frame_key] = {
                        "analysis": vars(fallback_analysis),  # Convert to dict
                        "prev_context": {
                            "prev_description": prev_analysis.description if prev_analysis else "None",
                            "prev_type": prev_analysis.anomaly_type if prev_analysis else "None",
                            "prev_score": round(prev_analysis.anomaly_score, 3) if prev_analysis else "None"
                        },
                        "clip_info": {
                            "center_frame": int(center_frame_idx),
                            "num_frames_in_clip": int(len(clip_frame_paths)) if clip_frame_paths else 0,
                            "error": str(e)
                        }
                    }
                    prev_analysis = fallback_analysis  # Update prev

                finally:
                    if temp_video_path:
                        self._cleanup_temp_video(temp_video_path)

            # Clear GPU cache periodically
            torch.cuda.empty_cache()

        self._save_scores(video_name, raw_scores)

    def _ensure_json_serializable(self, obj: Any) -> Any:
        """Recursively convert numpy types to JSON-serializable types with 3 decimal precision."""
        if isinstance(obj, dict):
            return {key: self._ensure_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return round(float(obj), 3)
        elif isinstance(obj, float):
            return round(obj, 3)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj

    def _save_scores(self, video_name: str, raw_scores: Dict[str, Any]) -> None:
        """Save raw scores to JSON file."""
        clean_raw_scores = self._ensure_json_serializable(raw_scores)
        output_path = self.output_scores_dir / f"{video_name}.json"
        
        try:
            with open(output_path, "w") as f:
                json.dump(clean_raw_scores, f, indent=4)
            logger.info(f"✅ Raw scores saved to {output_path}")
        except Exception as e:
            logger.error(f"❌ Error saving scores for {video_name}: {e}")


def load_model(model_path: str = "DAMO-NLP-SG/VideoLLaMA3-7B") -> Tuple[Any, Any]:
    """Load VideoLLaMA3 model and processor."""
    try:
        logger.info("Loading VideoLLaMA3 model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": "cuda:0"},
            torch_dtype=torch.bfloat16,
            enable_gradient_checkpointing=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        logger.info("✅ Model loaded successfully")
        return model, processor
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise


def run(
    root_path: str,
    annotationfile_path: str,
    batch_size: int,
    frame_interval: int,
    output_scores_dir: str,
    resume: bool,
    pathname: str,
    imagefile_template: str,
    fps: float,
    fps_create: float,
    fps_vlama: float,
    clip_duration: float,
    max_new_tokens: int,
    num_jobs: int,
    job_id: int

) -> None:
    """Main execution function."""
    # Load model
    model, processor = load_model()

    # Initialize detector
    device = "cuda:0"
    detector = VideoAnomalyDetector(
        model=model,
        processor=processor,
        device=device,
        root_path=root_path,
        output_scores_dir=output_scores_dir,
        batch_size=batch_size,
        frame_interval=frame_interval,
        fps=fps,
        fps_create=fps_create,
        fps_vlama=fps_vlama,
        clip_duration=clip_duration,
        imagefile_template=imagefile_template,  # Fixed: correct param name
        max_new_tokens=max_new_tokens
    )

    # Load and split video list
    with open(annotationfile_path, 'r') as f:
        video_list = [VideoRecord(x.strip().split(), root_path) for x in f]
    
    video_list = np.array_split(video_list, num_jobs)[job_id]
    
    if resume:
        video_list = find_unprocessed_videos(video_list, output_scores_dir, pathname)
      
    logger.info(f"Processing {len(video_list)} videos")

    # Process each video
    for video in video_list:
        try:
            detector.generate_raw_anomaly_scores(video)
        except Exception as e:
            logger.error(f"Failed to process video {video.video_name}: {e}")
            continue


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video Anomaly Raw Score Generator using VideoLLaMA3")
    parser.add_argument("--root_path", type=str, required=True, 
                       help="Path to frames directory")
    parser.add_argument("--annotationfile_path", type=str, required=True,
                       help="Path to annotation file")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing")
    parser.add_argument("--frame_interval", type=int, default=16,
                       help="Interval between frames")
    parser.add_argument("--output_scores_dir", type=str, required=True,
                       help="Output directory for raw scores")
    parser.add_argument("--resume", action="store_true",
                       help="Resume processing from unprocessed videos")
    parser.add_argument("--pathname", type=str, default="*.json",
                       help="Pattern for processed files")
    parser.add_argument("--imagefile_template", type=str, default="{:06d}.jpg",
                       help="Template for image filenames")
    parser.add_argument("--fps", type=float, required=True,
                       help="Original frames per second for calculations")
    parser.add_argument("--fps_create", type=float, default=1.0,
                       help="FPS for creating template video clips")
    parser.add_argument("--fps_vlama", type=float, default=1.0,
                       help="FPS for VideoLLaMA3 processing")
    parser.add_argument("--clip_duration", type=float, default=10.0,
                       help="Duration of each clip in seconds")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                       help="Maximum new tokens for generation")
    parser.add_argument("--num_jobs", type=int, default=1,
                       help="Number of parallel jobs")
    parser.add_argument("--job_id", type=int, default=0,
                       help="Job ID for parallel processing")
    

    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        root_path=args.root_path,
        annotationfile_path=args.annotationfile_path,
        batch_size=args.batch_size,
        frame_interval=args.frame_interval,
        output_scores_dir=args.output_scores_dir,
        resume=args.resume,
        pathname=args.pathname,
        imagefile_template=args.imagefile_template,
        fps=args.fps,
        fps_create=args.fps_create,
        fps_vlama=args.fps_vlama,
        clip_duration=args.clip_duration,
        max_new_tokens=args.max_new_tokens,
        num_jobs=args.num_jobs,
        job_id=args.job_id
    )
