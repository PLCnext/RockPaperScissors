import logging
import os
from threading import Lock
from typing import Any
from typing import Iterator
from typing import Literal

import cv2
import einops as eo
import numpy as np
from pypylon import pylon

from .event import TimedEvent
from .models import MODELS
from .processors import apply_processors
from .processors import BaseProcessor
from .processors import FPSAnnotator
from .processors import ModelAnnotator
from .settings import get_settings

logger = logging.getLogger('rps')

TL_FACTORY = pylon.TlFactory.GetInstance()
SETTINGS = get_settings()

_get_attributes = Literal[
    'AcquisitionFrameRate',
    'BlackLevel',
    'DigitalShift',
    'ExposureTime',
    'Gain',
    'Gamma',
    'BslLightDeviceBrightness',
    # 'BandwidthReserveMode',
]
_set_attributes = Literal[
    'AcquisitionFrameRate',
    'AcquisitionFrameRateEnable',
    'BlackLevel',
    'DigitalShift',
    'ExposureTime',
    'Gain',
    'Gamma',
    'BslLightDeviceBrightness',
    # 'BandwidthReserveMode',
]

_attribute_alias = {
    'BandwidthReserve': 'GevSCBWR',
}

_rotate_cv2 = {
    90: '90_CLOCKWISE',
    180: '180',
    270: '90_COUNTERCLOCKWISE',
}

MODEL = MODELS[SETTINGS.MODEL]
logger.info(f'PYLON_CAMEMU {os.environ.get("PYLON_CAMEMU")}')


class CameraException(Exception):
    """Exception thrown when an error is encountered with the camera that the
    user show know of.
    """


class Camera:
    """Wrapper around pylon.InstantCamera.

    Args:
        name (str): Friendly name of the camera.
    """

    _instances: dict[str, 'Camera'] = {}
    _lock: Lock = Lock()
    GET = _get_attributes
    SET = _set_attributes

    def __init__(self, name: str):
        self.name = name
        self.device = self._find_device(name)
        self.processors: list[BaseProcessor] = []

        self.capture_event = TimedEvent(SETTINGS.AFTER_CAPTURE_TIMER)
        self.show_model_view: bool = False
        self.rotation: int = int(SETTINGS.CAMERA_ROTATION)
        self._raw_img: np.ndarray = np.empty(0)
        self._img: np.ndarray = np.empty(0)
        self._timer: float = 0

    def open(self):
        """Opens the camera."""
        if not self.device.IsOpen():
            try:
                self.device.Open()
            except pylon.RuntimeException as e:
                raise CameraException(
                    f'Could not open camera "{self.name}". Make sure that the '
                    'camera is available and not used by another application.'
                ) from e

            # EMULATION SETTINGS
            if 'Emulation' in self.name:
                self.device.ImageFilename = (
                    SETTINGS.STATIC_DIR
                    / 'emulation'
                    / self.name.split('(')[-1][:-1]
                ).as_posix()
                self.device.ImageFileMode = 'On'
                self.device.TestImageSelector = 'Off'
                self.device.WidthMax = 1920
                self.device.HeightMax = 1080
                self.device.PixelFormat = (
                    'BGR8Packed'
                    if SETTINGS.CAMERA_PIXEL_FORMAT == 'BGR8'
                    else SETTINGS.CAMERA_PIXEL_FORMAT
                )
                self.device.AcquisitionFrameRateEnable = True
                self.device.AcquisitionFrameRate = 20

            else:
                self.device.PixelFormat = SETTINGS.CAMERA_PIXEL_FORMAT
                self.device.ExposureTime = SETTINGS.CAMERA_EXPOSURE_TIME
                self.device.DigitalShift = SETTINGS.CAMERA_DIGITAL_SHIFT
                self.device.Gamma = SETTINGS.CAMERA_GAMMA

                if SETTINGS.CAMERA_RING_LIGHT:
                    try:
                        self.device.BslLightControlMode = 'On'
                        self.device.BslLightControlEnumerateDevices.Execute()
                        self.device.BslLightDeviceSelector = 'Device1'
                        self.device.BslLightDeviceOperationMode = 'On'
                        self.device.BslLightDeviceBrightness = (
                            SETTINGS.CAMERA_RING_LIGHT_BRIGHTNESS
                        )
                    except pylon.GenericException as e:
                        logger.error(e)

            # set image format (cropped to center position)
            # we can set this in the camera or do it ourself
            # the later option consumes more bandwidth (good for this usecase)

            if SETTINGS.MODE_MAX_SIZE:
                self.device.Width = self.device.Width.Max
                self.device.Height = self.device.Height.Max
            else:
                self.device.Width = SETTINGS.CAMERA_WIDTH
                self.device.Height = SETTINGS.CAMERA_HEIGHT
                self.device.OffsetX = (
                    self.device.Width.Max // 2 - SETTINGS.CAMERA_WIDTH // 2
                )
                self.device.OffsetY = (
                    self.device.Height.Max // 2 - SETTINGS.CAMERA_HEIGHT // 2
                )

        self._setup_processors()

        if not self.device.IsGrabbing():
            self.device.StartGrabbing()

    def _setup_processors(self):
        """Initializes the processor pipeline."""
        bpp = int(self.device.PixelSize.Value.replace('Bpp', ''))
        img_shape = (self.device.Width.Value, self.device.Height.Value)

        self.processors = [
            ModelAnnotator(MODEL, color=(0, 0, 0)),
            FPSAnnotator(
                bpp=bpp,
                bw_unit=SETTINGS.BANDWIDTH_UNIT,
                img_shape=img_shape,
            ),
        ]

    def close(self):
        """Closes the camera."""
        if self.device.IsGrabbing():
            self.device.StopGrabbing()

        if self.device.IsOpen():
            self.device.Close()

        del self._instances[self.name]

    def stream(self) -> Iterator[bytes]:
        """
        Yields:
            Iterator[bytes]: Yields an iterator over the frames.
        """
        self.open()

        try:
            while self.device.IsGrabbing():
                img = self._get_image()
                yield from self._send_image(img)
        except pylon.GenericException as e:
            logger.error(e)
            raise CameraException(
                'Failed to process the camera stream.'
            ) from e
        finally:
            self.close()

    def _get_image(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Returns the latest image.
        """
        if not self.capture_event.is_set():
            with self.device.RetrieveResult(
                1000,
                pylon.TimeoutHandling_ThrowException,
            ) as res:
                if res.GrabSucceeded():
                    img: np.ndarray = res.Array

                    if self.rotation > 0:
                        degree = _rotate_cv2[self.rotation]
                        img = cv2.rotate(img, getattr(cv2, f'ROTATE_{degree}'))

                    if SETTINGS.CAMERA_PIXEL_FORMAT == 'BayerRG8':
                        img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    img = center_crop(
                        img,
                        (SETTINGS.CAMERA_WIDTH, SETTINGS.CAMERA_HEIGHT),
                    )
                    self._raw_img = img.copy()
                    self._img = apply_processors(img, self.processors)
                else:
                    logger.error(
                        f'Error: {res.ErrorCode}, {res.ErrorDescription}'
                    )

        else:
            # show captured image during event
            if self.capture_event.flag.get():
                self._img = apply_processors(
                    self._raw_img.copy(),
                    self.processors,
                    capture=True,
                )

        return self._img

    def _send_image(self, img: np.ndarray) -> Iterator[bytes]:
        """Prepares an image to send via JPEG stream.

        Args:
            img (np.ndarray): Image.

        Yields:
            Iterator[bytes]: Returns the bytes of a jpeg image.
        """
        if self.show_model_view:
            # processes image to model view
            img = MODEL.preprocess(img)[0].squeeze(0)
            img = eo.rearrange(img, 'C H W -> H W C') * 255
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        _, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type:image/jpeg\r\n'
            b'Content-Length: ' + f'{len(frame)}'.encode() + b'\r\n'
            b'\r\n' + frame + b'\r\n'
        )

    def _find_device(self, name: str) -> pylon.InstantCamera:
        """Finds a camera by ``name``.

        Args:
            name (str): Name of the camera.

        Raises:
            ValueError: Raised if no camera matches ``name``.

        Returns:
            pylon.InstantCamera: Returns an instance of the camera.
        """
        device = next(
            iter(
                [
                    device
                    for device in TL_FACTORY.EnumerateDevices()
                    if device.GetFriendlyName() == name
                ]
            ),
            None,
        )

        if device is None:
            raise ValueError(f"Camera '{name}' not found.")

        camera = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateDevice(device)
        )

        return camera

    @classmethod
    def get_available(cls) -> list[str]:
        """Returns a list of available cameras.

        Raises:
            NoDevicesError: Thrown if no devices are available.

        Returns:
            list[str]: Returns a list of available cameras.
        """
        devices = TL_FACTORY.EnumerateDevices()
        return [device.GetFriendlyName() for device in devices]

    @classmethod
    def get_instance(cls, name: str, **kwargs) -> 'Camera':
        """
        Args:
            name (str): Name of the camera.

        Returns:
            Camera: Returns the instance of the camera.
        """
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = Camera(name, **kwargs)

        return cls._instances[name]

    def get(self, attribute: _get_attributes) -> dict[str, Any]:
        """Gets the value and characteristics of ``attribute``.

        Args:
            attribute (_get_attributes): Name of the attribute.

        Returns:
            dict[str, Any]: Returns a mapping of the attribute.
        """
        _attribute = _attribute_alias.get(attribute, attribute)
        try:
            field = getattr(self.device, _attribute)

            enabled = (
                getattr(self.device, f'{_attribute}Enable').Value
                if f'{_attribute}Enable' in dir(self.device)
                else None
            )

            return {
                'value': getattr(field, 'Value'),
                'unit': getattr(field, 'Unit'),
                'min': getattr(field, 'Min'),
                'max': getattr(field, 'Max'),
                'enabled': enabled,
                'type': type(field).__name__,
            }
        except pylon.GenericException as e:
            logger.error(e)
            return {
                'value': 0,
                'unit': 'NA',
                'min': 0,
                'max': 0,
                'enabled': None,
                'type': 'IInteger',
            }

    def _get_enumeration(
        self,
        field: pylon.pypylon.genicam.IEnumeration,
    ) -> dict[str, Any]:
        return {
            'value': field.GetCurrentEntry().Symbolic,
            'entries': [e.Symbolic for e in field.GetEntries()],
            'unit': '',
            'enabled': None,
            'type': type(field).__name__,
        }

    def set(self, attribute: _set_attributes, value: Any):
        """Sets ``attribute`` to ``value``.

        Args:
            attribute (_set_attributes): Name of the attribute.
            value (Any): Value.

        Raises:
            ValueError: Raised if value could not be assigned.
        """
        _attribute = _attribute_alias.get(attribute, attribute)
        try:
            field = getattr(self.device, _attribute)
            field.SetValue(value)
        except pylon.GenericException as e:
            raise CameraException(
                f"Could not assign '{value}' to property '{attribute}'."
            ) from e


def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2]
    return crop_img
