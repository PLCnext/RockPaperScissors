from .base import BaseProcessor
from .base import FloatArray
from .fps import FPSAnnotator
from .image import ImageAnnotator
from .image import Position
from .model import ModelAnnotator


__all__ = [
    'BaseProcessor',
    'ImageAnnotator',
    'Position',
    'FPSAnnotator',
    'ModelAnnotator',
]


def apply_processors(
    img: FloatArray,
    processors: list[BaseProcessor],
    capture: bool = False,
) -> FloatArray:
    """Sequentially applies `processors` to `img`.

    Args:
        img (FloatArray): Image to process.
        processors (list[BaseProcessor]): List of processors.
        capture (bool): If True, then image is processed for a game result.

    Returns:
        FloatArray: Returns the processed image.
    """

    for processor in processors:
        img = processor(img, capture=capture)

    return img
