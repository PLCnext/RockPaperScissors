import logging
import os
import time
from typing import Annotated
from typing import Any
from typing import get_args
from typing import Literal

import jinja_partials
import shortuuid
from fastapi import Body
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi import status
from fastapi.exceptions import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .camera import Camera
from .opcua import get_opc_client
from .settings import get_settings
from .settings import OPCUASettings

logger = logging.getLogger('rps')
logger.setLevel(logging.INFO)

SETTINGS = get_settings()

templates = Jinja2Templates(directory=SETTINGS.STATIC_DIR / 'templates')
jinja_partials.register_starlette_extensions(templates)


def random_uuid(max_len: int = 5) -> str:
    """Generates a random uuid.

    Args:
        max_len (int, optional): Maximum length of uuid. Can be shorter.
          Defaults to 5.

    Returns:
        str: Returns a random uuid.
    """
    return shortuuid.uuid()[:max_len]


# add custom functions, use as {{ random_uuid() }} in template
templates.env.globals['random_uuid'] = random_uuid

app = FastAPI(
    title='RockPaperScissors',
    description='Demo application for SPS 2023.',
)


app.mount(
    '/static',
    StaticFiles(directory=SETTINGS.STATIC_DIR),
    name='static',
)


@app.get('/favicon.ico', include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(os.path.join(SETTINGS.STATIC_DIR, 'img/favicon.ico'))


@app.middleware('http')
async def add_process_time_header(request: Request, call_next):
    _start = time.time()
    response = await call_next(request)
    _end = time.time()
    response.headers['X-Process-Time'] = str(_end - _start)
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> Response:
    """Exception handler for validation requests.

    Args:
        exc (RequestValidationError): Error.

    Returns:
        Response: Returns a notification.
    """
    if 'HX-Request' in request.headers:
        responses: list[Response] = []
        for error in exc.errors():
            responses.append(
                render_toast(
                    message=f"{error['msg']} (input: '{error['input']}) "
                    f"for {error['loc']}",
                    type='danger',
                    delay=10000,
                )
            )

        body = '\r\n'.join([r.body.decode(r.charset) for r in responses])
        return HTMLResponse(
            body,
            headers={'HX-Reswap': 'none'},
        )

    raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, exc.errors())


@app.exception_handler(HTTPException)
async def unicorn_http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> Response:
    """Converts HTTPExceptions to a notification.

    Args:
        exc (HTTPException): Exception.

    Returns:
        Response: Returns a notification.
    """
    if 'HX-Request' in request.headers:
        return render_toast(
            message=f'{exc.status_code} - {exc.detail}',
            type='danger',
            delay=10000,
            headers={'HX-Reswap': 'none'},
        )

    raise exc


@app.exception_handler(Exception)
async def unicorn_general_exception_handler(
    request: Request,
    exc: Exception,
) -> Response:
    """Converts Exceptions to a notification.

    Args:
        exc (Exception): Exception.

    Returns:
        Response: Returns a notification.
    """
    return render_toast(
        message=str(exc),
        type='danger',
        delay=10000,
    )


@app.get('/')
async def root(
    request: Request,
) -> Response:
    """Returns the main page.

    Returns:
        HTMLResponse: Returns the main page.
    """
    return templates.TemplateResponse(
        'index.html.jinja',
        {
            'request': request,
        },
    )


@app.get('/cameras')
async def cameras(request: Request, camera: str | None = None) -> Response:
    """Returns a list of available cameras.

    Returns:
        HTMLResponse: Returns a list of available cameras.
    """
    cameras = Camera.get_available()

    return templates.TemplateResponse(
        'fragments/cameras.html.jinja',
        context={
            'request': request,
            'cameras': cameras,
            **({'selected': camera} if camera in cameras else {}),
        },
    )


@app.get('/cameras/{name}')
async def camera(
    request: Request,
    name: str,
    action: Literal['open', 'close'] = 'open',
) -> Response:
    """Returns an embedding of the camera stream.

    Args:
        name (str): Name of the camera.
        action ("open", "close"): Action. Default: "open".

    Returns:
        HTMLResponse: Returns an embedding of the camera stream.
    """
    camera = Camera.get_instance(name)

    if action == 'close':
        camera.close()
        return render_toast(
            f'Closed the camera "{camera}".',
            type='info',
            delay=5000,
        )

    camera.open()
    return templates.TemplateResponse(
        'fragments/video.html.jinja',
        context={
            'request': request,
            'url': request.url_for('camera_stream', name=name),
            'name': name,
            'message': f"Camera '{name}' selected.",
        },
    )


@app.get('/cameras/{name}/stream')
async def camera_stream(request: Request, name: str) -> StreamingResponse:
    """Opens or closes the a stream of the camera feed.

    Args:
        name (str): Name of the camera.
        action (Literal['open', 'close]): Action.

    Returns:
        StreamingResponse: Returns video stream.
    """
    camera = Camera.get_instance(name)
    camera.show_model_view = False

    return StreamingResponse(
        camera.stream(),
        media_type='multipart/x-mixed-replace; boundary=frame',
    )


@app.post('/cameras/{name}/view')
async def camera_view(
    request: Request,
    name: str,
) -> Response:
    """Toggles between model and camera view.

    Args:
        name (str): Name of the camera.

    Returns:
        Response: Returns a toast.
    """
    camera = Camera.get_instance(name)
    camera.show_model_view = not camera.show_model_view

    return render_toast(
        f'{"Enabled" if camera.show_model_view else "Disabled"} the model '
        'view.',
        type='success',
        delay=2000,
    )


@app.post('/cameras/{name}/rotate')
async def camera_rotation(
    request: Request,
    name: str,
) -> Response:
    camera = Camera.get_instance(name)
    camera.rotation = (camera.rotation + 90) % 360

    return render_toast(
        message=f'Set rotation to {camera.rotation}°.',
        type='success',
        delay=2000,
    )


@app.post('/cameras/{name}/play')
async def camera_play(request: Request, name: str) -> None:
    camera = Camera.get_instance(name)
    camera.capture_event.set()


@app.get('/cameras/{name}/settings')
async def settings(request: Request, name: str):
    camera = Camera.get_instance(name)
    fields = {field: camera.get(field) for field in get_args(Camera.GET)}

    return templates.TemplateResponse(
        'fragments/settings.html.jinja',
        context={
            'request': request,
            'camera': name,
            'fields': fields,
        },
    )


@app.get('/opcua')
async def get_opcua(request: Request) -> Response:
    """
    Returns:
        Response: Returns the current OPC UA settings.
    """
    opcua = SETTINGS.OPCUA.model_dump() if SETTINGS.OPCUA is not None else {}
    return templates.TemplateResponse(
        'fragments/opcua.html.jinja',
        context={
            'request': request,
            **opcua,
        },
    )


@app.post('/opcua')
async def change_opcua(
    request: Request,
    opcua: Annotated[OPCUASettings, Body()],
) -> Response:
    """Updates the OPC UA settings.

    Args:
        opcua (Annotated[OPCUASettings, Body): New settings.

    Returns:
        Response: Returns the new settings.
    """
    old_settings = (
        SETTINGS.OPCUA.model_copy() if SETTINGS.OPCUA is not None else None
    )
    SETTINGS.OPCUA = opcua
    client = get_opc_client()

    try:
        await client.update(opcua)
        await client.setup_client()
        if client._client is not None:
            await client._client.check_connection()
    except TimeoutError:
        SETTINGS.OPCUA = old_settings
        await client.update(old_settings)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            'Could not reach server. No change has been made.',
        )

    return templates.TemplateResponse(
        'fragments/opcua.html.jinja',
        context={
            'request': request,
            **SETTINGS.OPCUA.model_dump(),
        },
    )


@app.get('/cameras/{name}/settings/{field}')
async def get_value(
    request: Request,
    name: str,
    field: Camera.GET,
) -> dict[str, Any]:
    """Gets the value of a field.

    Args:
        name (str): Name of the camera.
        field (Camera.GET): Name of attribute.

    Returns:
        dict[str, Any]: Returns a mapping of the attribute characteristics.
    """
    return Camera.get_instance(name).get(field)


@app.post('/cameras/{name}/settings')
async def set_value(
    request: Request,
    name: str,
    fields: dict[str, int | float | bool],
) -> HTMLResponse:
    """Change camera settings.

    Args:
        name (str): Name of the camera.
        fields (dict[str, int  |  float  |  bool]): Mapping of attribute to
          value.

    Raises:
        HTTPException: Raised if attribute does not exist or value is invalid.

    Returns:
        HTMLResponse: Returns notifications.
    """
    if invalid := set(fields.keys()) - set(get_args(Camera.SET)):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f'Properties {list(invalid)} are invalid. '
            f'Available: {Camera.SET}.',
        )

    responses: list[Response] = []
    for field, value in fields.items():
        try:
            Camera.get_instance(name).set(field, value)
            responses.append(
                render_toast(
                    message=f"Changed {field} to '{value}'.",
                    type='success',
                    delay=2000,
                )
            )
        except ValueError:
            responses.append(
                render_toast(
                    message=f"Could not change {field} to '{value}'.",
                    type='danger',
                )
            )

    body = '\r\n'.join([r.body.decode(r.charset) for r in responses])
    return HTMLResponse(body)


def render_toast(
    message: str,
    *,
    type: Literal['info', 'success', 'warning', 'danger'] = 'info',
    delay: float = 5000,
    headers: dict[str, str] = {},
    status_code: int = status.HTTP_200_OK,
) -> Response:
    """Renders a notification.

    Args:
        message (str): Message to display.
        type (Literal['info', 'success', 'warning', 'danger'], optional):
          Type of message. Defaults to "info".
        delay (float, optional): Time to show the notification.
          Defaults to 5000.
        headers (dict[str, str], optional): Additional response header.
          Defaults to {}.

    Returns:
        HTMLResponse: Returns the notification.
    """
    return templates.TemplateResponse(
        'fragments/toast.html.jinja',
        headers=headers,
        status_code=status_code,
        context={
            'request': {},
            'message': message,
            'type': type,
            'delay': delay,
        },
    )
