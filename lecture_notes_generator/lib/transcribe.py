from typing import Callable, Iterator, TextIO, TypedDict

import whisper
from whisper.utils import write_txt, write_vtt

from .utils import cached


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


@cached
def transcribe(
    path, model="large", language="en", initial_prompt=None
) -> tuple[str, list[FullSegment]]:
    path = str(path)

    model = whisper.load_model(model)

    # When "verbose" is False, it displays a progress bar with tqdm
    result = model.transcribe(
        path,
        verbose=False,
        language=language,
        initial_prompt=initial_prompt,
    )

    return result["text"], result["segments"]


if __name__ == "__main__":

    from .paths import TESTS_DIR

    path = TESTS_DIR / "video.mp4"
    text, segments = transcribe(
        path,
        model="tiny",
        initial_prompt="Hello. Welcome to the Science education channel.",
    )

    print(text)
    print(segments)
