import asyncio
from abc import abstractmethod, ABC
from pathlib import Path
from time import monotonic
from typing import NamedTuple, AsyncGenerator, TypeVar, Generator, Mapping, Sequence, Union, Optional

import attrs
import cattrs
import cv2
import numpy as np
import piexif
from PIL import Image
from aiohttp import ClientSession, ClientResponse

_T_co = TypeVar("_T_co", covariant=True)

DEFAULT_FOV = 90
DEFAULT_POVS_ARGS = (8, 0), (6, 45)


class Location(NamedTuple):
    lat: float
    lng: float

    def __str__(self):
        return f"{self.lat},{self.lng}"

    @classmethod
    def of(cls, data: Union[Mapping[str, float], Sequence[float]], *_) -> 'Location':
        if isinstance(data, Location):
            return data
        if isinstance(data, Mapping):
            return cls(data['lat'], data['lng'])
        return cls(data[0], data[1])


cattrs.register_structure_hook(Location, Location.of)
cattrs.register_unstructure_hook(Location, lambda x: {'lat': x.lat, 'lng': x.lng})


def float_to_deg(n: float):
    d, md = divmod(n, 1)
    m, ms = divmod(md * 60, 1)
    return (int(d), 1), (int(m), 1), (int(ms * 60 * 10000), 10000)


def image_from_bytes(image_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def read_image(path: Path):
    return image_from_bytes(path.read_bytes())


def write_image(image: np.ndarray, path: Path, mkdir: bool = False, location: Location = None):
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    if not any((location,)):
        return
    img = Image.open(path)
    params = {}
    exif_dict = {}
    if 'exif' in img.info:
        exif_dict.update(piexif.load(img.info['exif']))
    if location is not None:
        exif_dict['GPS'] = {
            piexif.GPSIFD.GPSLatitudeRef: 'N',
            piexif.GPSIFD.GPSLatitude: float_to_deg(location.lat),
            piexif.GPSIFD.GPSLongitudeRef: 'E',
            piexif.GPSIFD.GPSLongitude: float_to_deg(location.lng),
        }
    if exif_dict:
        params['exif'] = piexif.dump(exif_dict)
    img.save(path, **params)


@attrs.define
class BaseImage(ABC):

    @property
    @abstractmethod
    def image(self) -> Optional[np.ndarray]:
        pass

    @property
    @abstractmethod
    def default_filename(self) -> str:
        pass

    @property
    def location(self) -> Optional[Location]:
        return

    def save(self, path: Path, mkdir: bool = False, error_ok: bool = False):
        if path.is_dir():
            path = path / self.default_filename
        image = self.image
        if image is None:
            if error_ok:
                return
            raise ValueError("Image has not been defined")
        write_image(image, path, mkdir, self.location)
        return path


class LimitedClientSession(ClientSession):
    def __init__(self, limit: int, per_secs: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._limit = limit
        self._per_secs = per_secs
        self._semaphore = asyncio.Semaphore(limit)
        self._finish_times = []

    def __await__(self):
        return self.__aenter__().__await__()

    async def _request(self, *args, **kwargs) -> ClientResponse:
        async with self._semaphore:
            if len(self._finish_times) >= self._limit:
                sleep_until = self._finish_times.pop(0)
                if sleep_until >= monotonic():
                    await asyncio.sleep(sleep_until - monotonic())
            res = await super()._request(*args, **kwargs)
            await res.read()
            self._finish_times.append(monotonic() + self._per_secs)
        return res


async def image_from_res(res: ClientResponse):
    # body = await res.read() doesn't work if connection is closed
    body = (await res.text('latin-1')).encode('latin-1')
    return image_from_bytes(body)


async def wrap_in_coro(data):
    return data


def sync_generator(agen: AsyncGenerator[_T_co, None], loop=None) -> Generator[_T_co, None, None]:
    if new_loop := loop is None:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _END_ITERATION = object()

    async def _next():
        try:
            return await async_generator.__anext__()
        except StopAsyncIteration:
            return _END_ITERATION

    value: _T_co
    async_generator = agen.__aiter__()
    while True:
        value = loop.run_until_complete(_next())
        if value is _END_ITERATION:
            break
        yield value
    if new_loop:
        loop.close()
