from pathlib import Path
from typing import Any, Iterable, List

def _as_path(x: Any) -> Path:
    """Return a Path for strings/Path/objects with .path."""
    p = getattr(x, "path", x)
    return Path(p)

def _basename_no_ext(x: Any) -> str:
    """Basename without extension for strings/Path/objects with .path."""
    return _as_path(x).stem

def find_unprocessed_videos(video_list: List[Any], output_dir: str | Path, pathname: str) -> List[Any]:
    """
    Given a list of videos (items may be strings, Paths, or objects with `.path`),
    return the sublist of videos that come AFTER the most recently processed one.
    `output_dir/glob(pathname)` should point to your processed outputs.
    """
    output_dir = Path(output_dir)
    processed_videos = set(output_dir.glob(pathname))

    if not processed_videos:
        return video_list  # nothing processed yet

    # Pick the *most recently modified* processed file, not lexicographically last
    try:
        last_processed = max(processed_videos, key=lambda p: p.stat().st_mtime)
    except FileNotFoundError:
        # If a file disappeared between glob and stat, fall back to lexicographic
        last_processed = max(processed_videos)

    last_name = last_processed.stem

    # Build comparable names from the original list (stem on each item)
    names = [_basename_no_ext(v) for v in video_list]

    # Find the index of the last processed item; if not found, treat as none processed
    try:
        last_idx = names.index(last_name)
    except ValueError:
        return video_list

    next_idx = last_idx + 1
    if next_idx >= len(video_list):
        return []
    return video_list[next_idx:]


def find_last_processed_video_index(video_list: Iterable[Any], last_processed_video_path: Path) -> int:
    """
    Keeps your helper, but make it resilient.
    Returns -1 if not found (callers can handle slicing from 0 or similar).
    """
    target = Path(last_processed_video_path).stem
    names = [_basename_no_ext(v) for v in video_list]
    try:
        return names.index(target)
    except ValueError:
        return -1

