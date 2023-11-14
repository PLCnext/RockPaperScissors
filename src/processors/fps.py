import time
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from .base import FloatArray
from .base import SETTINGS
from .image import ImageAnnotator
from .image import Position

__all__ = ['FPSAnnotator']


@dataclass
class FPSCounter:
    """Stores the FPS counter."""

    counter: int = 0
    fps: float = 0.0
    start_time: float = field(default_factory=time.time)

    def update(self) -> float:
        """Updates the FPS counter.

        Returns:
            float: Returns the current FPS.
        """
        self.counter += 1
        if ((ct := time.time()) - self.start_time) > 1:
            self.fps = self.counter / (ct - self.start_time)
            self.counter = 0
            self.start_time = ct

        return self.fps


class FPSAnnotator(ImageAnnotator):
    """Adds an FPS counter to an image.

    Args:
        img_shape (tuple[int, int]): Original shape of the image.
        bpp (int): Bits per pixel.
        bw_unit (['MB', 'Mbit]): Unit of the bandwidth.
        color (tuple[int, int, int]): Text color. Default: (0, 255, 0).
        position (Position): Text position. Default: top-left.
        fontScale (float, optional): Text size.
            Defaults to SETTINGS.ANNOTATION_FONT_SCALE.
        thickness (float, optional): Text thickness.
            Defaults to SETTINGS.ANNOTATION_THICKNESS_SCALE.
    """

    def __init__(
        self,
        img_shape: tuple[int, int],
        bpp: int,
        bw_unit: Literal['MB', 'Mbit'] = 'Mbit',
        color: tuple[int, int, int] = (0, 255, 0),
        position: Position = Position.TL,
        fontScale: float = SETTINGS.ANNOTATION_FONT_SCALE,
        thickness: float = SETTINGS.ANNOTATION_THICKNESS_SCALE,
    ):
        super().__init__(
            color=color,
            position=position,
            fontScale=fontScale,
            thickness=thickness,
        )

        self.img_shape = img_shape
        self.bpp = bpp
        self.bw_unit = bw_unit

        self._counter = FPSCounter()

    def __call__(
        self,
        img: FloatArray,
        *,
        capture: bool = False,
    ) -> FloatArray:
        if capture:
            return img

        fps = self._counter.update()
        bandwidth = self._calculate_bandwidth(fps)

        return self._putText(
            img,
            text=f'FPS: {fps:.0f} ({bandwidth:.1f} {self.bw_unit}/s)',
        )

    def _calculate_bandwidth(
        self,
        fps: float,
    ) -> float:
        """Calculates the bandwidth per second (received.)

        Args:
            fps (int): Frames per second.

        Returns:
            float: Returns the bandwidth per second.
        """
        bandwidth = fps * self.img_shape[0] * self.img_shape[1] * self.bpp

        if self.bw_unit == 'MB':
            bandwidth /= 8

        return bandwidth / (1000**2)
