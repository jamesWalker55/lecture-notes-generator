import inspect
import json
import os
import re
from collections import deque
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TextIO


def parse_duration(text):
    """Parse a duration like '09:12.160' into a number"""
    try:
        match text.split(":"):
            case h, m, s:
                h = int(h)
                m = int(m)
                s = float(s)
                return h * 60**2 + m * 60 + s
            case m, s:
                m = int(m)
                s = float(s)
                return m * 60 + s
            case s:
                return float(s)
    except ValueError as e:
        raise ValueError(f"Invalid duration string: {text}")


def get_default_value(func, param: str):
    sig = inspect.signature(func)
    if param not in sig.parameters:
        raise ValueError(f"Parameter {param!r} doesn't exist in function {func!r}")
    param = sig.parameters[param]
    value = param.default
    if value == inspect._empty:
        raise ValueError(f"Parameter {param!r} in function {func!r} has no default value")
    return value


def each_cons(it, n, pad_first_iteration=False):
    """
    - pad_first_iteration:
      When the number of values is less than 'n', by default it immediately stops
      iterating, e.g. `list(each_cons([1,2], 5)) == []`. When this is set to True,
      ensure there is always a first iteration, and pad the remaining places with
      `None`, e.g. `list(each_cons([1,2], 5)) == [(1, 2, None, None, None)]`.
    """
    # convert it to an iterator
    it = iter(it)

    # insert first n items to a list first
    deq = deque()
    for _ in range(n):
        try:
            deq.append(next(it))
        except StopIteration:
            if pad_first_iteration:
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


def file_cache(
    path_func: Callable[..., Path],
    dump_func: Callable[[Any, Path], None],
    load_func: Callable[[Path], Any],
):
    """Retrieve values from a file if the path exists, otherwise save output of the function to file"""

    def wrapper(func):
        @wraps(func)
        def wrapped_func(*args, skip_loading=False, **kwargs):
            cache_path = Path(path_func(*args, **kwargs))
            if not skip_loading and cache_path.exists():
                print(f"Loading from cached file: {cache_path}")
                try:
                    return load_func(cache_path)
                except Exception as e:
                    print("Failed to load from cached file.")
                    raise e

            value = func(*args, **kwargs)

            dump_func(value, cache_path)

            return value

        return wrapped_func

    return wrapper
