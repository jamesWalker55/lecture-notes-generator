import json
from typing import Literal, NamedTuple

from .transcribe import Segment
from .utils import each_cons
from .video import get_snapshots


class Scene(NamedTuple):
    cut: int
    segments: list[Segment]


class RenderScene(NamedTuple):
    type: Literal["subtitles", "slideshow"]
    cuts: list[int]
    segments: list[Segment]


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

        current_scene = Scene(start_frame, [])
        scenes.append(current_scene)

        while len(segments) > 0:
            seg = segments[-1]
            if start_time <= seg["start"] < end_time:
                current_scene.segments.append(segments.pop())
            else:
                break

    # if there are any remaining segments, add them to the final scene
    final_scene = Scene(scene_change_frames[-1], segments.copy())
    scenes.append(final_scene)

    return scenes


def group(scenes: list[Scene]):
    render_scenes: list[RenderScene] = []
    current_slideshow_scene = None
    for sc in scenes:
        if len(sc.segments) == 0:
            if current_slideshow_scene is None:
                current_slideshow_scene = RenderScene("slideshow", [], [])
                render_scenes.append(current_slideshow_scene)
            current_slideshow_scene.cuts.append(sc.cut)
        else:
            if current_slideshow_scene is not None:
                current_slideshow_scene = None
            render_scenes.append(
                RenderScene(
                    "subtitles",
                    [sc.cut],
                    sc.segments,
                )
            )
    return render_scenes


if __name__ == "__main__":
    import cv2

    from .paths import TESTS_DIR
    from .transcribe import transcribe
    from .utils import _cache_default_handler
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
    for frame, img in zip(scene_changes, get_snapshots(video, scene_changes)):
        cv2.imwrite("c/f{:06d}.jpg".format(frame), img)

    with open("c/text.json", "w", encoding="utf8") as f:
        json.dump(scenes, f, indent=2, default=_cache_default_handler)
