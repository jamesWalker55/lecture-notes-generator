import json
from pathlib import Path

import cv2
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .paths import TEMPLATES_DIR, TESTS_DIR
from .scenes import Scene, generate_scenes
from .transcribe import transcribe
from .video import detect_scene_changes, export_snapshots, get_fps

ENV = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(),
)


def render_scenes(scenes, title, html_path, frames_dir):
    html_path = Path(html_path)
    frames_dir = Path(frames_dir)

    # make frame_dir a relative path to the html, because the browser isn't good at absolute paths
    frames_dir = frames_dir.relative_to(html_path.parent)

    # render the site
    template = ENV.get_template("main.html")
    html = template.render(scenes=scenes, frames_dir=frames_dir, title=title)

    # save the site
    with open(html_path, "w", encoding="utf8") as f:
        f.write(html)


if __name__ == "__main__":

    def temp(
        path, transcribe_model="large", transcribe_language="en", initial_prompt=None
    ):
        fps = get_fps(path)

        # some arguments
        path = Path(path)
        html_path = path.with_suffix(".html")
        snapshot_dir = path.parent / f"frames_{path.stem}"

        # run whisper on video to get subtitles
        segments = transcribe(
            path,
            model=transcribe_model,
            language=transcribe_language,
            initial_prompt=initial_prompt,
        )

        # detect scene cuts in the video
        scene_cuts = detect_scene_changes(path)
        # save screenshots of scene cuts to path
        export_snapshots(path, scene_cuts, snapshot_dir)

        # group them into scenes
        scenes = generate_scenes(scene_cuts, segments, fps)

        render_scenes(scenes, path.stem, html_path, snapshot_dir)

    video_path = TESTS_DIR / "video.mp4"
    temp(video_path, transcribe_model="tiny")
