import cv2

from .paths import CACHE_DIR
from .utils import cached_value, each_cons


def every_frame(cap):
    yield from every_n_frames(cap, 1)


def every_n_frames(cap, n):
    # start at frame 0
    count = 0

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        try:
            yield frame
        except Exception as e:
            cap.release()
            raise e

        count += n  # advance by n frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)

    cap.release()


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.signal

    from .paths import TESTS_DIR

    video_path = TESTS_DIR / "video.mp4"
    video = cv2.VideoCapture(str(video_path))
    diff = cached_value(
        lambda: frames_absolute_diff(video),
        CACHE_DIR / "video_main_cache.json",
    )

    # plot result in a graph
    plt.plot(diff)

    # find and plot the peaks
    peaks = scipy.signal.find_peaks(diff, distance=30, threshold=12)
    for x in peaks[0]:
        plt.plot(x, diff[x], "yo")

    # show the plot
    plt.show()
