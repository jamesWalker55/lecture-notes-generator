from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from whisper import _MODELS
from whisper.tokenizer import LANGUAGES

from .lib.render import render_scenes
from .lib.scenes import generate_scenes
from .lib.transcribe import transcribe
from .lib.utils import get_default_value
from .lib.video import detect_scene_changes, get_snapshots, export_snapshots, get_fps


def get_parser():
    # fmt: off

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("paths", nargs="+", type=Path, help="path to the video files to transcribe")

    gp = parser.add_argument_group("Whisper subtitles transcription")
    gp.add_argument("--model", "-m", choices=_MODELS.keys(), default=get_default_value(transcribe, 'model'), help="the model used to transcribe the audio")
    gp.add_argument("--language", "-lang", choices=LANGUAGES.keys(), default=get_default_value(transcribe, 'language'), help="the language of the audio")
    gp.add_argument("--initial-prompt", "-p", type=str, default=get_default_value(transcribe, 'initial_prompt'), help="initial prompt for whisper transcription")

    gp = parser.add_argument_group("Scene cut detection")
    gp.add_argument("--diff-height", "-dh", type=float, default=get_default_value(detect_scene_changes, 'height'), help="required height of peaks (absolute height)")
    gp.add_argument("--diff-threshold", "-dt", type=float, default=get_default_value(detect_scene_changes, 'threshold'), help="required height of peaks (relative to neighboring samples)")
    gp.add_argument("--diff-distance", "-dd", type=float, default=get_default_value(detect_scene_changes, 'distance'), help="required horizontal distance between peaks")
    gp.add_argument("--diff-prominence", "-dp", type=float, default=get_default_value(detect_scene_changes, 'prominence'), help="required prominence of peaks, \"how much a peak stands out from the surrounding baseline of the signal\"")
    gp.add_argument("--diff-width", "-dw", type=float, default=get_default_value(detect_scene_changes, 'width'), help="required width of peaks")
    gp.add_argument("--diff-wlen", "-dl", type=float, default=get_default_value(detect_scene_changes, 'wlen'), help="used for calculating peak prominence")
    gp.add_argument("--diff-rel-height", "-dr", type=float, default=get_default_value(detect_scene_changes, 'rel_height'), help="used for calculating peak width")
    gp.add_argument("--diff-plateau-size", "-ds", type=float, default=get_default_value(detect_scene_changes, 'plateau_size'), help="\"Required size of the flat top of peaks in samples\"")

    # fmt: on

    return parser


def cli():
    args = get_parser().parse_args()
    whisper_kwargs = {
        "model": args.model,
        "language": args.language,
        "initial_prompt": args.initial_prompt,
    }
    scene_kwargs = {
        "height": args.diff_height,
        "threshold": args.diff_threshold,
        "distance": args.diff_distance,
        "prominence": args.diff_prominence,
        "width": args.diff_width,
        "wlen": args.diff_wlen,
        "rel_height": args.diff_rel_height,
        "plateau_size": args.diff_plateau_size,
    }
    for p in args.paths:
        process_path(p, whisper_kwargs, scene_kwargs)


def process_path(path, whisper_kwargs: dict, scene_kwargs: dict):
    path = Path(path)

    # get fps of path
    fps = get_fps(path)

    # some arguments
    path = Path(path)
    html_path = path.with_suffix(".html")
    snapshot_dir = path.parent / f"frames_{path.stem}"

    # run whisper on video to get subtitles
    segments = transcribe(path, **whisper_kwargs)
    print(f"Transcribed {len(segments)} segments")

    # detect scene cuts in the video
    scene_cuts = detect_scene_changes(path, **scene_kwargs)
    print(f"Detected {len(scene_cuts)} scene cuts")
    # save screenshots of scene cuts to path
    snapshots = get_snapshots(path, scene_cuts)
    export_snapshots(snapshots, snapshot_dir)

    # group them into scenes
    scenes = generate_scenes(scene_cuts, segments, fps)

    render_scenes(scenes, path.stem, html_path, snapshot_dir)
