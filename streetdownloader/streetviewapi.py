import re
from os import getenv

import attrs
import cattrs
import numpy as np
from aiohttp import ClientResponse
from dotenv import load_dotenv

from streetdownloader.common import BaseImage, Location, LimitedClientSession, image_from_res

GOOGLE_SV_API = 'https://maps.googleapis.com/maps/api/streetview'
MAX_API_REQ_PER_MIN = 32000

load_dotenv()
GOOGLE_API_KEY = getenv("GOOGLE_API_KEY")
GOOGLE_API_SIGNATURE = getenv("GOOGLE_API_SIGNATURE")


@attrs.define
class Metadata:
    status: str
    copyright: str = None
    date: str = None
    location: Location = None
    pano_id: str = None


@attrs.define
class RequestParams:
    pano: str = None
    heading: float = None  # 0 < heading < 360
    fov: float = 90  # 0 < fov < 120
    pitch: float = 0  # -90 < pitch < 90


@attrs.define
class View(BaseImage):
    image: np.ndarray = None
    params: RequestParams = None
    metadata: Metadata = None


def _prompt_google_api_key():
    from getpass import getpass
    global GOOGLE_API_KEY
    print("Missing Google API key. Insert GOOGLE_API_KEY in a '.env' file.")
    print("Get your API key at: https://developers.google.com/maps/documentation/streetview/get-api-key")
    api_key = getpass('Google API key:')
    assert re.match("^AIzaSy[A-Za-z0-9_-]{33}$", api_key), "Invalid API key"
    GOOGLE_API_KEY = api_key


class StreetViewAPI(LimitedClientSession):
    def __init__(self, limit: int = MAX_API_REQ_PER_MIN, per_secs: float = 60, *args, **kwargs):
        super().__init__(limit, per_secs, *args, **kwargs)

    async def _api_get(self, url: str, params: dict) -> ClientResponse:
        if GOOGLE_API_KEY is None:
            _prompt_google_api_key()
        _p = {**params, 'size': '640x640', 'key': GOOGLE_API_KEY}
        if GOOGLE_API_SIGNATURE:
            _p['signature'] = GOOGLE_API_SIGNATURE
        async with self.get(url, params=_p) as res:
            return res

    async def request_metadata(self, location: Location) -> Metadata:
        _p = {'location': f'{location[0]:.7f},{location[1]:.7f}'}
        res = await self._api_get(GOOGLE_SV_API + '/metadata', _p)
        return cattrs.structure(await res.json(), Metadata)

    async def request_image(self, location: Location = None, metadata: Metadata = None, heading: float | None = None,
                            fov: float | None = None, pitch: float | None = None) -> View | None:
        if metadata is None and location is not None:
            metadata = await self.request_metadata(location)
        if metadata is None or metadata.status != 'OK':
            return
        rp = RequestParams(metadata.pano_id, heading, fov, pitch)
        res = await self._api_get(GOOGLE_SV_API, {k: v for k, v in attrs.asdict(rp).items() if v is not None})
        if res.ok:
            return View(await image_from_res(res), rp, metadata)
