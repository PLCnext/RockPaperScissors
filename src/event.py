import time
from threading import Event


class TimedEvent(Event):
    """Timed event."""

    def __init__(self, duration: float) -> None:
        super().__init__()
        self.duration = duration
        self._time = 0.0
        self.flag = Flag()

    def set(self) -> None:
        self._time = time.time()
        self.flag.set()
        return super().set()

    def is_set(self) -> bool:
        if self._flag and (time.time() - self._time) > self.duration:
            self.clear()

        return super().is_set()


class Flag:
    def __init__(self):
        self._flag = False

    def set(self):
        self._flag = True

    def get(self) -> bool:
        if old := self._flag:
            self._flag = False

        return old
