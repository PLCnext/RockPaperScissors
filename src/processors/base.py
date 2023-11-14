from abc import ABC
from abc import abstractmethod

import numpy as np
import numpy.typing as npt

from ..settings import get_settings

SETTINGS = get_settings()

__all__ = ['BaseProcessor']

FloatArray = npt.NDArray[np.float_]


class BaseProcessor(ABC):
    @abstractmethod
    def __call__(
        self,
        img: FloatArray,
        *,
        capture: bool = False,
    ) -> FloatArray:
        """Processes an Image.

        Args:
            img (FloatArray): Image.
            capture (bool): If true, then the image is captured to be
              processed for a game result.


        Returns:
            FloatArray: Returns the processed image.
        """
