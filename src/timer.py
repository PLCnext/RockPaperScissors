import time
import typing as T
from functools import wraps

returnT = T.TypeVar('returnT')


def timer(
    function: T.Callable[..., returnT]
) -> T.Callable[..., tuple[returnT, float]]:
    @wraps(function)
    def wrapper(*args, **kwargs) -> tuple[returnT, float]:
        _start = time.time()
        result = function(*args, **kwargs)
        _end = time.time()

        return result, _end - _start

    return wrapper
