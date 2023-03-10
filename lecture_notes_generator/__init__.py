from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from whisper import _MODELS
from whisper.tokenizer import LANGUAGES

from .lib.render import render_scenes
from .lib.scenes import generate_scenes
from .lib.transcribe import transcribe
from .lib.utils import get_default_value
from .lib.video import (
    detect_scene_changes,
    _export_snapshots,
    get_snapshots,
    get_fps,
    get_snapshots,
)


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

    gp = parser.add_argument_group("Cache")
    gp.add_argument("--retranscribe", "-rt", action="store_true", help="ignore any existing subtitle files and re-transcribe the video")
    gp.add_argument("--rediff", "-rd", action="store_true", help="ignore any existing absdiff files and rescan the video for frame differences")

    gp = parser.add_argument_group("Other")
    gp.add_argument("--snapshot-delay", "-sd", type=int, default=15, help="wait a certain amount of frames before taking a snapshot for the scene cut")
    gp.add_argument("--transcribe-only", "-t", action="store_true", help="only transcribe the video and skip all other steps")
    gp.add_argument("--skip-notes", "-s", action="store_true", help="skip generating the notes output and snapshots")

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
        "snapshot_delay": args.snapshot_delay,
    }
    other_kwargs = {
        "snapshot_delay": args.snapshot_delay,
        "retranscribe": args.retranscribe,
        "rediff": args.rediff,
        "transcribe_only": args.transcribe_only,
        "skip_notes": args.skip_notes,
    }
    for p in args.paths:
        print(f"\nProcessing file: {p}")
        process_path(p, whisper_kwargs, scene_kwargs, other_kwargs)


def process_path(path, whisper_kwargs: dict, scene_kwargs: dict, other_kwargs: dict):
    path = Path(path)

    # get fps of path
    fps = get_fps(path)

    # some arguments
    path = Path(path)
    html_path = path.with_suffix(".html")
    snapshot_dir = path.parent / f"frames_{path.stem}"

    # run whisper on video to get subtitles
    segments = transcribe(
        path, **whisper_kwargs, skip_loading=other_kwargs["retranscribe"]
    )

    # return early if we're only transcribing
    print(f"Transcribed {len(segments)} segments")

    if other_kwargs["transcribe_only"]:
        return

    # detect scene cuts in the video
    scene_cuts = detect_scene_changes(
        path, **scene_kwargs, skip_loading=other_kwargs["rediff"]
    )
    print(f"Detected {len(scene_cuts)} scene cuts")

    # return early if we're skipping notes generation
    if other_kwargs["skip_notes"]:
        return

    # save screenshots of scene cuts to path
    snapshot_dir.mkdir(exist_ok=True)
    _export_snapshots(scene_cuts, snapshot_dir)

    # group them into scenes
    scene_cuts_f = [x[0] for x in scene_cuts]
    scenes = generate_scenes(scene_cuts_f, segments, fps)

    render_scenes(scenes, path.stem, html_path, snapshot_dir)
