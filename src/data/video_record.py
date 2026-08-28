import os
from typing import List, Union


class VideoRecord:
    """This class represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index. If the video has multiple labels,
                the fourth element is a comma-separated string of label indices.
    """

    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def path(self) -> str:
        return self._path

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive

    @property
    def start_frame(self) -> int:
        return int(self._data[1])

    @property
    def end_frame(self) -> int:
        return int(self._data[2])

    @property
    def label(self) -> Union[int, List[int]]:
        return [int(label_id) for label_id in self._data[3].split(",")]












# from pathlib import Path
# import os
# # ##

# run VL ober this for score anomaly on videollama3
# class VideoRecord:
#     def __init__(self, row, root_path):
#         self.video_name = row[0]  # e.g., Abuse028_x264
#         self.start_frame = int(row[1])  # e.g., 0
#         self.end_frame = int(row[2])  # e.g., 1411
#         self.label = int(row[3])  # e.g., 0
#         self.root_path = os.path.normpath(root_path)  # e.g., /mnt/.../ucf_crim_data/frames
#         self.path = os.path.normpath(os.path.join(self.root_path, self.video_name))  # e.g., /mnt/.../ucf_crim_data/frames/Abuse028_x264
#         frame_dir = Path(self.path)
#         print(frame_dir)
#         if not frame_dir.exists():
#             print(f"Error: Frame directory {frame_dir} does not exist for {self.video_name}")
#             self.num_frames = 0
#         else:
#             frame_files = sorted(frame_dir.glob("*.jpg"))
#             self.num_frames = len(frame_files)
#             expected_num_frames = self.end_frame - self.start_frame + 1
#             if self.num_frames != expected_num_frames:
#                 print(f"Warning: num_frames mismatch for {self.video_name}: Expected {expected_num_frames}, found {self.num_frames}")
#             if self.num_frames == 0:
#                 print(f"Warning: No frames found in {frame_dir} for {self.video_name}")