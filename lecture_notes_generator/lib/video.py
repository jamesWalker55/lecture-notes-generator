import json
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
import scipy.signal
from tqdm import tqdm

from .utils import each_cons, file_cache


def _every_frame(cap):
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    with tqdm(total=length, initial=start_pos) as pbar:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            try:
                yield frame
            except Exception as e:
                cap.release()
                raise e

            pbar.update(1)

    cap.release()


def _json_default_handler(o):
    if isinstance(o, (np.int64, np.uint32)):
        return int(o)
    raise TypeError(o.__class__)


def _frames_absolute_diff_dump(obj, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, default=_json_default_handler)


def _frames_absolute_diff_load(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def _frames_absolute_diff_path(video_path):
    video_path = Path(video_path)
    return video_path.with_name(f"absdiff_{video_path.stem}.json")


@file_cache(
    _frames_absolute_diff_path, _frames_absolute_diff_dump, _frames_absolute_diff_load
)
def frames_absolute_diff(path):
    if isinstance(path, Path):
        cap = cv2.VideoCapture(str(path))
    else:
        cap = cv2.VideoCapture(path)

    result = []
    count = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for prev_frame, frame in each_cons(_every_frame(cap), 2):
        diff = cv2.absdiff(prev_frame, frame)

        # normalize diff according to video size, so that it is consistent across different resolutions
        normalized_diff = diff.sum() / width / height
        # # normalise diff to 0..1, it was a value 0..(256**3)
        # normalized_diff = normalized_diff / 256**3

        result.append(normalized_diff)

        count += 1

    return result


def get_fps(path):
    if isinstance(path, Path):
        cap = cv2.VideoCapture(str(path))
    else:
        cap = cv2.VideoCapture(path)

    return float(cap.get(cv2.CAP_PROP_FPS))


def _unique_frames(path, frames: List[int], threshold: int = 10, delay: int=0):
    """Given a video and a bunch of frame numbers, return frames that are unique from the previous frame"""
    if isinstance(path, Path):
        cap = cv2.VideoCapture(str(path))
    else:
        cap = cv2.VideoCapture(path)

    uniq_scenes = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    snapshots = get_snapshots(path, frames, delay=delay)
    scene_img = None  # the frame at the beginning of the current scene

    for f, img in snapshots:
        if scene_img is None:
            scene_img = img
            uniq_scenes.append((f, img))
        else:
            diff = cv2.absdiff(scene_img, img)
            normalized_diff = diff.sum() / width / height

            if normalized_diff > threshold:
                scene_img = img
                uniq_scenes.append((f, img))

    return uniq_scenes


def detect_scene_changes(
    path,
    height=None,
    threshold=3,
    distance=20,
    prominence=None,
    width=None,
    wlen=None,
    rel_height=0.5,
    plateau_size=None,
    skip_loading=False,
    snapshot_delay=None,
) -> List[Tuple[int, Any]]:
    """
    Return the frame numbers where scene changes occur in a given video. All kwargs are for the
    `scipy.signal.find_peaks()` function:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy-signal-find-peaks

    - height: required height of peaks (absolute height)
    - threshold: required height of peaks (relative to neighboring samples)
    - distance: required horizontal distance between peaks
    - prominence: required prominence of peaks, "how much a peak stands out from the surrounding baseline of the signal"
    - width: required width of peaks
    - wlen: used for calculating peak prominence
    - rel_height: used for calculating peak width
    - plateau_size: "Required size of the flat top of peaks in samples"
    """
    diff = frames_absolute_diff(path, skip_loading=skip_loading)

    # find and plot the peaks
    peaks = scipy.signal.find_peaks(
        diff,
        height=height,
        threshold=threshold,
        distance=distance,
        prominence=prominence,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
        plateau_size=plateau_size,
    )
    peak_frames = [0, *(x + 1 for x in peaks[0])]
    return _unique_frames(path, peak_frames, delay=snapshot_delay)


def get_snapshots(path, frames: List[int], delay: int = 0):
    if isinstance(path, Path):
        cap = cv2.VideoCapture(str(path))
    else:
        cap = cv2.VideoCapture(path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    snapshots = []

    for f in frames:
        delayed_f = min(f + delay, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, delayed_f)
        ok, frame = cap.read()
        # try reading the next few frames
        for _ in range(10):
            if ok:
                break
            ok, frame = cap.read()
        # still not ok, raise error this time
        if not ok:
            raise RuntimeError(f"Failed to get frame #{delayed_f} of video.")
        snapshots.append((f, frame))

    return snapshots


def _get_comparison_snapshots(path, frames: List[int]):
    if isinstance(path, Path):
        cap = cv2.VideoCapture(str(path))
    else:
        cap = cv2.VideoCapture(path)

    snapshots = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for f in frames:
        if f == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            black_frame = np.zeros((height, width, 3), np.uint8)
            ok, zero_frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to get first frame of video.")
            snapshots.append((0, black_frame, zero_frame))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
            ok, before_frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to get frame #{f - 1} of video.")
            ok, after_frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to get frame #{f} of video.")
            snapshots.append((f, before_frame, after_frame))

    return snapshots


def _export_snapshots(snapshots: List[Tuple[int, Any]], output_folder):
    for i, img in snapshots:
        img_path = output_folder / f"f{i:06d}.jpg"
        cv2.imwrite(str(img_path), img)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from .paths import TESTS_DIR

    video_path = TESTS_DIR / "video.mp4"
    scene_changes = detect_scene_changes(
        str(video_path),
        distance=30,
        threshold=12,
        _cache_name=video_path,
    )

    comparisons = _get_comparison_snapshots(str(video_path), scene_changes)
    count = 0
    for _, before, after in tqdm(comparisons):
        joined = np.concatenate((before, after), axis=1)
        cv2.imwrite("c/cut{:d}.jpg".format(count), joined)
        count += 1

    # show the plot
    plt.show()
