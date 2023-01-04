import cv2
import numpy as np
from tqdm import tqdm

from .utils import cached, each_cons


def every_frame(cap):
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


def every_n_frames(cap, n):
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


@cached
def frames_absolute_diff(cap):
    result = []
    count = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for prev_frame, frame in each_cons(every_frame(cap), 2):
        diff = cv2.absdiff(prev_frame, frame)

        # normalize diff according to video size, so that it is consistent across different resolutions
        normalized_diff = diff.sum() / width / height
        # # normalise diff to 0..1, it was a value 0..(256**3)
        # normalized_diff = normalized_diff / 256**3

        result.append(normalized_diff)

        count += 1

    return result


def get_comparison_snapshots(cap, frames: list[int]):
    frames = sorted(frames)
    snapshots = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if 0 in frames:
        frames.remove(0)
        black_frame = np.zeros((height, width, 3), np.uint8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, zero_frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to get first frame of video.")
        snapshots.append((0, black_frame, zero_frame))

    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, before_frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to get frame #{f - 1} of video.")
        ok, after_frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to get frame #{f} of video.")
        snapshots.append((f, before_frame, after_frame))

    return snapshots


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.signal

    from .paths import TESTS_DIR

    video_path = TESTS_DIR / "video.mp4"
    video = cv2.VideoCapture(str(video_path))
    diff = frames_absolute_diff(video, _cache_name=video_path)

    # plot result in a graph
    plt.plot(diff)

    # find and plot the peaks
    peaks = scipy.signal.find_peaks(diff, distance=30, threshold=12)
    scene_changes = [0, *(x + 1 for x in peaks[0])]

    for x in scene_changes:
        plt.plot(x, diff[x], "yo")

    comparisons = get_comparison_snapshots(video, scene_changes)
    count = 0
    for _, before, after in tqdm(comparisons):
        joined = np.concatenate((before, after), axis=1)
        cv2.imwrite("c/cut{:d}.jpg".format(count), joined)
        count += 1

    # show the plot
    plt.show()
