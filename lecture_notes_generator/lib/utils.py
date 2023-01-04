import json
import os
from collections import deque

import numpy


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
