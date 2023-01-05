from .transcribe import Segment
from .utils import each_cons

# [0] - the scene cut
# [1] - list of text segments
Scene = tuple[int, list[Segment]]


def pair(
    scene_change_frames: list[int],
    segments: list[Segment],
    video_fps: int | float,
):
    scene_change_frames = sorted(scene_change_frames)
    # sort segments in reverse order
    segments = sorted(segments, key=lambda x: x["start"], reverse=True)

    scenes: list[Scene] = []

    for start_frame, end_frame in each_cons(scene_change_frames, 2):
        start_time = start_frame / video_fps
        end_time = end_frame / video_fps

        current_scene: Scene = (start_frame, [])
        scenes.append(current_scene)

        while len(segments) > 0:
            seg = segments[-1]
            if start_time <= seg["start"] < end_time:
                current_scene[1].append(segments.pop())
            else:
                break

    # if there are any remaining segments, add them to the final scene
    final_scene: Scene = (scene_change_frames[-1], segments.copy())
    scenes.append(final_scene)

    return scenes


if __name__ == "__main__":
    import cv2

    from .paths import TESTS_DIR
    from .transcribe import transcribe
    from .video import detect_scene_changes

    video_path = TESTS_DIR / "video.mp4"
    video = cv2.VideoCapture(str(video_path))

    scene_changes = detect_scene_changes(
        video,
        distance=30,
        threshold=12,
        _cache_name=video_path,
    )

    text, segments = transcribe(
        video_path,
        model="tiny",
        initial_prompt="Hello. Welcome to the Science education channel.",
    )

    fps = float(video.get(cv2.CAP_PROP_FPS))

    scenes = pair(scene_changes, segments, fps)
    print(scenes)
