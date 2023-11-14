import json
import typing as T

import cv2
import einops as eo
import mlnext
import numpy as np
from scipy.special import softmax

from ..settings import get_settings
from ..timer import timer
from .base import ModelWrapper

# import torch
# from torchvision import transforms

__all__ = ['CNN']

SETTINGS = get_settings()
MODEL_NAME = f'cnn_v{SETTINGS.MODEL_VERSION}'

with open(SETTINGS.STATIC_DIR / 'model' / 'classes.json', 'r') as f:
    CLASS_NAMES: dict[str, str] = json.load(f)


class Transforms:
    """Implements the preprocessing pipeline from torchvision with opencv."""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(
            img,
            (SETTINGS.IMAGE_HEIGHT, SETTINGS.IMAGE_WIDTH),
            interpolation=cv2.INTER_AREA,
        )
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.float32) / 255.0  # scale to [0, 1]

        img = (img - 0.6) / 0.2  # z-norm (x - mean) / std

        img = eo.rearrange(img, 'H W -> 1 1 H W')

        return img


class CNN(ModelWrapper):
    def __init__(self):
        super().__init__(MODEL_NAME)
        self.transforms = Transforms()
        # self.transforms = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.ConvertImageDtype(dtype=torch.float32),
        #         transforms.Grayscale(num_output_channels=1),
        #         transforms.Resize(
        #             (SETTINGS.IMAGE_HEIGHT, SETTINGS.IMAGE_WIDTH),
        #             antialias=True,
        #         ),
        #         transforms.Normalize(
        #             mean=[0.6],
        #             std=[0.2],
        #             # mean=[0.8497, 0.8211, 0.8112],
        #             # std=[0.2472, 0.2887, 0.3032],
        #         ),
        #     ]
        # )

    @timer
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # img = self.transforms(img).numpy()
        # return eo.rearrange(img, 'C H W -> 1 C H W')
        return self.transforms(img)

    @timer
    def postprocess(self, outputs: dict[str, np.ndarray]) -> dict[str, T.Any]:
        scores = softmax(outputs['scores'])
        labels = mlnext.eval_softmax(scores).reshape(-1)
        classes = list(map(lambda label: CLASS_NAMES.get(str(label)), labels))

        return {
            'scores': scores.tolist(),
            'labels': labels.tolist(),
            'classes': classes,
        }
