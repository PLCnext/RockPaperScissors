import math
from enum import Enum

import cv2

from .base import BaseProcessor
from .base import FloatArray
from .base import SETTINGS

__all__ = [
    'Position',
    'ImageAnnotator',
]


class Position(str, Enum):
    """Defines a Position on the image."""

    TL = 'top-left'
    TR = 'top-right'
    BL = 'bottom-left'
    BR = 'bottom-right'

    def get(self, shape: tuple[int, ...]) -> tuple[int, int]:
        """
        Returns:
            tuple[int, int]: Returns the coordinates for the specific position.
        """
        match self.value:
            case Position.TL:
                return (0, 0)
            case Position.TR:
                return (shape[0], 0)
            case Position.BL:
                return (0, shape[1])
            case Position.BR:
                return (shape[0], shape[1])

        raise ValueError('Invalid option.')


class ImageAnnotator(BaseProcessor):
    """Base class for processor with image annotations.

    Args:
        color (tuple[int, int, int]): Text color.
        position (Position): Text position.
        fontScale (float, optional): Text size.
            Defaults to SETTINGS.ANNOTATION_FONT_SCALE.
        thickness (float, optional): Text thickness.
            Defaults to SETTINGS.ANNOTATION_THICKNESS_SCALE.
    """

    def __init__(
        self,
        color: tuple[int, int, int],
        position: Position,
        fontScale: float = SETTINGS.ANNOTATION_FONT_SCALE,
        thickness: float = SETTINGS.ANNOTATION_THICKNESS_SCALE,
    ):
        super().__init__()
        self.position = position
        self.color = color
        self.fontScale = fontScale
        self.thickness = thickness

    def _putText(
        self,
        img: FloatArray,
        text: str,
        fontFace: int = cv2.FONT_HERSHEY_DUPLEX,
    ) -> FloatArray:
        """Puts ``text`` on ``img``. Wrapper for cv2.putText.

        Args:
            img (FloatArray): Image to modify.
            text (str): Text to insert.
            fontFace (int): Font to use.

        Returns:
            FloatArray: Returns the image.
        """
        height, width, _ = img.shape
        fontScale = min(width, height) * self.fontScale
        thickness = math.ceil(min(width, height) * self.thickness)

        (w, h), _ = cv2.getTextSize(text, fontFace, fontScale, thickness)

        # calculate position
        x, y = self.position.get(img.shape)
        x = max(5, min(x, width - w))
        y = max(h + 5, min(y, height - h))

        cv2.putText(
            img,
            text,
            org=(x, y),
            fontFace=fontFace,
            fontScale=fontScale,
            color=self.color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        return img
