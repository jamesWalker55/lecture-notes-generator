import json
import os
import re
from collections import deque
from functools import wraps

import numpy

from .paths import CACHE_DIR


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


def _cache_default_handler(o):
    if isinstance(o, (numpy.int64, numpy.uint32)):
        return int(o)
    raise TypeError(o.__class__)


def cached_value(func, path):
    # Try to load existing cache
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf8") as f:
                return json.load(f)
    except Exception as e:
        pass

    # Make sure cache parent folder exists
    try:
        os.mkdir(os.path.join(path, ".."))
    except FileExistsError:
        pass

    # Call the function
    value = func()

    # Save the value to cache
    with open(path, "w", encoding="utf8") as f:
        json.dump(value, f, default=_cache_default_handler)

    return value


def sanitize_text_for_path(text):
    return "".join([c for c in text if re.match(r"\w", c)])


def cached(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        output_stem = None

        if "_cache_name" in kwargs:
            if kwargs["_cache_name"] is not None:
                output_stem = sanitize_text_for_path(str(kwargs["_cache_name"]))

            del kwargs["_cache_name"]

        if output_stem is None:
            args_str = "_".join(str(x) for x in args)
            kwargs_str = "_".join(f"{k}={v}" for k, v in kwargs.items())
            output_stem = sanitize_text_for_path(f"{args_str}_{kwargs_str}")

        output_stem = f"{func.__name__}_{output_stem}"
        output_path = (CACHE_DIR / output_stem).with_suffix(".json")

        return cached_value(
            lambda: func(*args, **kwargs),
            output_path,
        )

    return wrapped_func
