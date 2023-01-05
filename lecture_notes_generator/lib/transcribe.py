import json
from pathlib import Path
from typing import Callable, Iterator, TextIO, TypedDict

import webvtt
import whisper
from whisper.utils import write_txt, write_vtt

from .utils import file_cache, parse_duration


class FullSegment(TypedDict):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class Segment(TypedDict):
    start: float
    end: float
    text: str


# Custom typing for functinos in whisper
write_txt: Callable[[Iterator[FullSegment], TextIO], None]
write_vtt: Callable[[Iterator[FullSegment], TextIO], None]


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
) -> list[Segment]:
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
