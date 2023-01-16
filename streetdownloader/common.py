import asyncio
from abc import abstractmethod, ABC
from pathlib import Path
from time import monotonic
from typing import NamedTuple, AsyncGenerator, TypeVar, Generator, Mapping, Sequence

import attrs
import cattrs
import cv2
import numpy as np
from aiohttp import ClientSession, ClientResponse

_T_co = TypeVar("_T_co", covariant=True)


class Location(NamedTuple):
    lat: float
    lng: float

    @classmethod
    def of(cls, data: Mapping[str, float] | Sequence[float], *_) -> 'Location':
        if isinstance(data, Location):
            return data
        if isinstance(data, Mapping):
            return cls(data['lat'], data['lng'])
        return cls(data[0], data[1])


cattrs.register_structure_hook(Location, Location.of)


@attrs.define
class BaseImage(ABC):

    @property
    @abstractmethod
    def image(self):
        pass

    def save(self, path: Path, mkdir: bool = False):
        if mkdir:
            path.parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite(str(path), self.image)
        ext = '.' + path.name.rsplit('.', 1)[-1]
        path.write_bytes(np.array(cv2.imencode(ext, self.image)[1]).tobytes())


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


async def image_from_res(res):
    # body = await res.read() doesn't work if connection is closed
    body = (await res.text('latin-1')).encode('latin-1')
    array = np.asarray(bytearray(body), dtype="uint8")
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


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
