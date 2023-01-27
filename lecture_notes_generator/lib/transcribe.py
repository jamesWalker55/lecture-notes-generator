from pathlib import Path
from typing import Iterator, List, TextIO, TypedDict

import webvtt
import whisper

from .utils import file_cache, parse_duration


class FullSegment(TypedDict):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class Segment(TypedDict):
    start: float
    end: float
    text: str


# methods copied from whisper
def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def write_txt(result: Iterator[FullSegment], f: TextIO) -> None:
    for segment in result:
        print(segment["text"].strip(), file=f, flush=True)


def write_vtt(result: Iterator[FullSegment], f: TextIO) -> None:
    print("WEBVTT\n", file=f)
    for segment in result:
        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=f,
            flush=True,
        )


def _transcribe_load(path: Path):
    segments = [
        {"start": parse_duration(x.start), "end": parse_duration(x.end), "text": x.text}
        for x in webvtt.read(path)
    ]
    txt_path = path.with_suffix(".txt")
    if not txt_path.exists():
        with open(txt_path, "w", encoding="utf8") as f:
            write_txt(segments, f)
    return segments


def _transcribe_dump(segments, path: Path):
    with open(path, "w", encoding="utf8") as f:
        write_vtt(segments, f)
    with open(path.with_suffix(".txt"), "w", encoding="utf8") as f:
        write_txt(segments, f)


def _transcribe_path(video_path, **kwargs):
    video_path = Path(video_path)
    return video_path.with_suffix(".vtt")


@file_cache(_transcribe_path, _transcribe_dump, _transcribe_load)
def transcribe(
    path, model="large", language="en", initial_prompt=None
) -> List[Segment]:
    path = str(path)

    model = whisper.load_model(model)

    # When "verbose" is False, it displays a progress bar with tqdm
    result = model.transcribe(
        path,
        verbose=False,
        language=language,
        initial_prompt=initial_prompt,
    )

    return result["segments"]


if __name__ == "__main__":

    from .paths import TESTS_DIR

    path = TESTS_DIR / "video.mp4"
    segments = transcribe(
        path,
        model="tiny",
        initial_prompt="Hello. Welcome to the Science education channel.",
    )

    print(segments)
