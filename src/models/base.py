import os
import typing as T
from abc import ABC
from abc import abstractmethod

import numpy as np
import onnxruntime as ort

from ..settings import get_settings
from ..timer import timer

SETTINGS = get_settings()

__all__ = ['ModelWrapper']


class ModelWrapper(ABC):
    """Common wrapper for ONNX models.

    Args:
        name (str): Name of the model (w/ or w/o extension).
    """

    def __init__(self, name: str) -> None:
        self.name, self.path = self._check_model(name)

        self.model, self._load_t = self.load_model()
        self.outputs = [out.name for out in self.model.get_outputs()]

    def _check_model(self, name: str) -> tuple[str, str]:
        """Checks whether the model exists on disk.

        Args:
            name (str): Name of the model (with or without extension).

        Raises:
            ValueError: Raised if the model does not exist.

        Returns:
            tuple[str, str]: Returns the name and path to the model.
        """
        if not name.endswith('.onnx'):
            name = f'{name}.onnx'

        path = SETTINGS.STATIC_DIR / 'model' / name
        if not os.path.isfile(path):
            raise ValueError(f'Model not found "{name}".')

        return name, path

    @timer
    def load_model(self) -> ort.InferenceSession:
        """Loads the model.

        Args:
            path (str): Path to the model.

        Returns:
            ort.InferenceSession: Returns the loaded session.
        """
        provider = f'{get_settings().EXECUTION_PROVIDER}ExecutionProvider'
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.add_session_config_entry(
            'session.set_denormal_as_zero', '1'
        )

        return ort.InferenceSession(
            self.path, providers=[provider], sess_options=sess_options
        )

    @timer
    @abstractmethod
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Subclasses should overwrite this method to perform preprocessing
        operations, e.g., as Resize and Crop.

        Args:
            img (np.ndarray): Image.

        Returns:
            np.ndarray: Returns the preprocessed image.
        """

    @timer
    @abstractmethod
    def postprocess(self, outputs: tuple[np.ndarray, ...]) -> dict[str, T.Any]:
        """Postprocesses the outputs of the model.

        Args:
            outputs (tuple[np.ndarray, ...]): A tuple of outputs.

        Returns:
            dict[str, T.Any]: Returns the results.
        """

    @timer
    def run(self, img: np.ndarray) -> dict[str, np.ndarray]:
        """Runs inference with ``img`` on the model.

        Args:
            img (np.ndarray): Image to classify.

        Returns:
            dict[str, np.ndarray]: Returns a mapping with the outputs.
        """
        outputs = self.model.run(None, {'input': img})
        return dict(zip(self.outputs, outputs))

    def predict(self, img: np.ndarray) -> dict[str, T.Any]:
        """Performs a prediction with the wrapped model.

        Args:
            img (np.ndarray): Image as array.

        Returns:
            T.Any: Returns the results.
        """
        img, _pre_t = self.preprocess(img)
        outputs, _run_t = self.run(img)
        results, _post_t = self.postprocess(outputs)

        return {
            **results,
            'timer': {
                'load': self._load_t,
                'pre': _pre_t,
                'run': _run_t,
                'post': _post_t,
                'total': (_pre_t + _run_t + _post_t),
            },
        }
