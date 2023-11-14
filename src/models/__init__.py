import typing as T

from ..settings import get_settings
from .base import ModelWrapper
from .cnn import CNN

SETTINGS = get_settings()

MODEL_CLASSES = {
    'cnn': CNN,
}

MODELS: dict[str, ModelWrapper] = {}

_model_names = tuple(MODEL_CLASSES.keys())
MODEL_NAMES = T.Literal[_model_names]  # type: ignore

REQUESTED_MODELS = list(map(str.lower, [SETTINGS.MODEL]))

for m in _model_names:
    if m in REQUESTED_MODELS:
        MODELS[m] = MODEL_CLASSES[m]()  # type: ignore
    else:
        raise ValueError(f'Model {m} not found.')
