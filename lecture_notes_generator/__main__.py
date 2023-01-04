import json
import os
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy
import scipy.signal


def every_frame(cap):
    yield from every_n_frames(cap, 1)


def every_n_frames(cap, n):
    # start at frame 0
    count = 0

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        try:
            yield frame
        except Exception as e:
            video.release()
            raise e

        count += n  # advance by n frames
        video.set(cv2.CAP_PROP_POS_FRAMES, count)

    video.release()


def each_cons(it, n):
    # convert it to an iterator
    it = iter(it)

    # insert first n items to a list first
    deq = deque()
    for _ in range(n):
        try:
            deq.append(next(it))
        except StopIteration:
            for _ in range(n - len(deq)):
                deq.append(None)
            yield tuple(deq)
            return

    yield tuple(deq)

    # main loop
    while True:
        try:
            val = next(it)
        except StopIteration:
            return
        deq.popleft()
        deq.append(val)
        yield tuple(deq)


def plot(data):
    plt.plot(data)
    plt.show()


DATA_PATH = "./data.json"

if not os.path.exists("./data.json"):
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    result = []
    count = 0

    video = cv2.VideoCapture(r"tests\video.mp4")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for prev_frame, frame in each_cons(every_frame(video), 2):
        # cv2.imwrite("a/frame{:d}.jpg".format(i), frame)
        # print("#", end="")

        diff = cv2.absdiff(prev_frame, frame)

        # normalize diff according to video size, so that it is consistent across different resolutions
        normalized_diff = diff.sum() / width / height
        # # normalise diff to 0..1, it was a value 0..(256**3)
        # normalized_diff = normalized_diff / 256**3

        result.append(normalized_diff)

        # cv2.imwrite("a/frame{:d}.jpg".format(count), diff)

        # mask = background_subtractor.apply(prev_frame)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # difference = cv2.absdiff(frame, mask)
        # sum_difference = difference.sum()
        # print(sum_difference)

        count += 1

    def converter(o):
        if isinstance(o, (numpy.int64, numpy.uint32)):
            return int(o)
        raise TypeError(o.__class__)

    with open(DATA_PATH, "w", encoding="utf8") as f:
        json.dump(result, f, default=converter)
else:
    with open(DATA_PATH, "r", encoding="utf8") as f:
        result = json.load(f)

# plot result in a graph
plt.plot(result)

# find and plot the peaks
peaks = scipy.signal.find_peaks(result, distance=30, threshold=15)
for x in peaks[0]:
    plt.plot(x, result[x], "yo")

# show the plot
plt.show()
