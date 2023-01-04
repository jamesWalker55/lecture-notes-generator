from typing import TypedDict

import whisper

from .utils import cached


class Segment(TypedDict):
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


@cached
def transcribe(path, model="large", language="en") -> tuple[str, list[Segment]]:
    path = str(path)

    model = whisper.load_model(model)

    # When "verbose" is False, it displays a progress bar with tqdm
    result = model.transcribe(path, verbose=False, language=language)

    return result["text"], result["segments"]


if __name__ == "__main__":

    from .paths import TESTS_DIR

    path = TESTS_DIR / "video.mp4"
    text, segments = transcribe(path, model="tiny")

    print(text)
    print(segments)
