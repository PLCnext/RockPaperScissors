import asyncio

from ..models import ModelWrapper
from ..opcua import get_opc_client
from .base import FloatArray
from .base import SETTINGS
from .image import ImageAnnotator
from .image import Position


class ModelAnnotator(ImageAnnotator):
    """Adds an model prediction to an image.

    Args:
        color (tuple[int, int, int]): Text color. Default:(0, 152, 161).
        position (Position): Text position. Default: top-left.
        fontScale (float, optional): Text size.
            Defaults to SETTINGS.ANNOTATION_FONT_SCALE.
        thickness (float, optional): Text thickness.
            Defaults to SETTINGS.ANNOTATION_THICKNESS_SCALE.
    """

    i = 101

    def __init__(
        self,
        model: ModelWrapper,
        sampling_factor: int = SETTINGS.MODEL_SAMPLING_FACTOR,
        color: tuple[int, int, int] = (255, 0, 0),
        position: Position = Position.BL,
        fontScale: float = SETTINGS.ANNOTATION_FONT_SCALE,
        thickness: float = SETTINGS.ANNOTATION_THICKNESS_SCALE,
    ):
        super().__init__(
            color=color,
            position=position,
            fontScale=fontScale,
            thickness=thickness,
        )

        self.model = model
        self.sampling_factor = sampling_factor

        self._counter: int = 0
        self._label: int = 0
        self._class: str = 'blank'
        self._score: float = 1.0
        self._total: float = 0.0

        self._opc_client = get_opc_client()

    def __call__(
        self,
        img: FloatArray,
        *,
        capture: bool = False,
    ) -> FloatArray:
        self._counter += 1
        if self._counter >= self.sampling_factor or capture:
            # cv2.imwrite(f'recorded/{type(self).i}.png', img)
            # type(self).i += 1
            self._predict(img)
            self._counter = 0

        if capture:
            asyncio.run(self._opc_client.write_value(self._label))

        return self._putText(
            img,
            text=f'{self._class} ({self._score*100:.1f}%) '
            f'[{self._total*1000:.0f}ms]',
        )

    def _predict(self, img: FloatArray):
        outputs = self.model.predict(img)

        self._label = outputs['labels'][0]
        self._class = outputs['classes'][0]
        self._score = max(outputs['scores'][0])
        self._total = outputs['timer']['total']
