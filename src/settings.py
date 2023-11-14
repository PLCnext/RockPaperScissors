from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Annotated
from typing import Literal

from pydantic import AfterValidator
from pydantic import BaseModel
from pydantic import DirectoryPath
from pydantic import Field
from pydantic import UrlConstraints
from pydantic_core import Url
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


@dataclass(slots=True)
class RequireParts:
    username: bool = False
    password: bool = False

    def __call__(self, v: Url) -> Url:
        for field in self.__slots__:
            if getattr(v, field) is None:
                raise ValueError(
                    f'{field.capitalize()} is required in url '
                    '"schema://username:password@host:port/path?'
                    'query#fragment".'
                )

        return v


OpcUaUrl = Annotated[
    Url,
    UrlConstraints(
        allowed_schemes=['opc.tcp', 'http', 'https'],
        host_required=True,
        default_port=4840,
    ),
    AfterValidator(RequireParts(username=True, password=True)),
]


class OPCUASettings(BaseModel):
    """Settings for OPC UA communication."""

    url: OpcUaUrl = Field(
        ...,
        description='Url of the form '
        '"[opc.tcp|http|https]://[user]:[pass]@[host]:[port]/".',
    )

    nodeid: str = Field(
        ...,
        description='NodeID of the form "ns=<ns>;s=<s>.',
        min_length=1,
    )


_static = Path(__file__).parent / 'static'


class Settings(BaseSettings):
    STATIC_DIR: DirectoryPath = Field(
        _static,
        description='Path to static dir.',
        frozen=True,
    )

    EXECUTION_PROVIDER: Literal['CPU'] = Field(
        'CPU',
        description='Execution provider to run the ONNX model. Default: CPU.',
    )

    CAMERA_WIDTH: int = Field(
        900,
        ge=1,
        description='Width of the camera image. Cropped from the orignal '
        'image. Use OffsetX and OffsetY to change the crop position.',
    )

    CAMERA_HEIGHT: int = Field(
        900,
        ge=1,
        description='Height of the camera image. Cropped from the orignal '
        'Use OffsetX and OffsetY to change the crop position.',
    )

    CAMERA_PIXEL_FORMAT: Literal['BayerRG8', 'BGR8'] = Field(
        'BayerRG8',
        description='Pixel format. Controls how colors are represented and '
        'with how many bits.',
    )

    CAMERA_EXPOSURE_TIME: float = Field(
        10000,
        description='Default value for the exposure time. Can be controlled '
        'on the frontend.',
    )

    CAMERA_DIGITAL_SHIFT: int = Field(
        0,
        description='Default value for the digital shift. Can be controlled '
        'on the frontend.',
    )

    CAMERA_GAMMA: float = Field(
        2,
        description='Default value for the gamma. Can be controlled on the '
        'frontend.',
    )

    CAMERA_ROTATION: Literal[0, 90, 180, 270, '0', '90', '180', '270'] = Field(
        180,
        description='Rotate the image by some degrees.',
    )

    CAMERA_RING_LIGHT: bool = Field(
        True,
        description='Whether to turn on the connected light.',
    )

    CAMERA_RING_LIGHT_BRIGHTNESS: int = Field(
        45, description='Ring light brigthness.'
    )

    BANDWIDTH_UNIT: Literal['MB', 'Mbit'] = Field(
        'Mbit',
        description='Unit to use for displayed bandwidth.',
    )

    MODE_MAX_SIZE: bool = Field(
        False,
        description='Whether to set the camera to send the maximum image '
        'size.',
    )

    IMAGE_WIDTH: int = Field(
        300,
        ge=1,
        description='Width of the images. Default: 300.',
    )

    IMAGE_HEIGHT: int = Field(
        300,
        ge=1,
        description='Height of the images. Default: 300.',
    )

    MODEL: Literal['cnn'] = Field(
        'cnn',
        description='Model to use. Default: ["cnn"].',
    )

    MODEL_VERSION: str = Field(
        '72',
        description='Model version.',
    )

    MODEL_SAMPLING_FACTOR: int = Field(
        5,
        description='Sampling factor for the model evaluation. '
        'This value sets the sampling rate to evaluate every n-th image. '
        'Default: 5 (every 5th frame).',
    )

    ANNOTATION_FONT_SCALE: float = Field(
        2e-3,
        description='Font scale for annotations. Default: 2e-3.',
    )

    ANNOTATION_THICKNESS_SCALE: float = Field(
        3e-3,
        description='Thickness scale for annotations. Default: 2e-3.',
    )

    AFTER_CAPTURE_TIMER: float = Field(
        5,
        description='Time in s to show the image after capture. Default: 5.',
    )

    OPCUA: OPCUASettings | None = Field(
        None,
        description='Settings for OPC UA connection.',
    )

    model_config = SettingsConfigDict(
        env_file=['.env', _static / 'env/.env'],
        extra='allow',
        case_sensitive=False,
        env_nested_delimiter='__',
    )


@cache
def get_settings() -> Settings:
    return Settings()
