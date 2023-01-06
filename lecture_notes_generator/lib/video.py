import json
from pathlib import Path

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


def _every_n_frames(cap, n):
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    countdown = 0
    with tqdm(total=length, initial=start_pos) as pbar:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            if countdown == 0:
                countdown = n - 1
                try:
                    yield frame
                except Exception as e:
                    cap.release()
                    raise e
            else:
                countdown -= 1

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
    return video_path.with_stem(video_path.stem + "_absdiff").with_suffix(".json")


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


def detect_scene_changes(
    path,
    height=None,
    threshold=None,
    distance=None,
    prominence=None,
    width=None,
    wlen=None,
    rel_height=0.5,
    plateau_size=None,
    _cache_name=None,
) -> list[int]:
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
    """
    diff = frames_absolute_diff(path)

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
    return [0, *(x + 1 for x in peaks[0])]


def get_snapshots(path, frames: list[int]):
    if isinstance(path, Path):
        cap = cv2.VideoCapture(str(path))
    else:
        cap = cv2.VideoCapture(path)

    snapshots = []

    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to get frame #{f} of video.")
        snapshots.append((f, frame))

    return snapshots


def get_comparison_snapshots(path, frames: list[int]):
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


def export_snapshots(path, frames: list[int], output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    snapshots = get_snapshots(path, frames)

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

    comparisons = get_comparison_snapshots(str(video_path), scene_changes)
    count = 0
    for _, before, after in tqdm(comparisons):
        joined = np.concatenate((before, after), axis=1)
        cv2.imwrite("c/cut{:d}.jpg".format(count), joined)
        count += 1

    # show the plot
    plt.show()
