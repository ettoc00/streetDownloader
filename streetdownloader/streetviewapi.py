import asyncio
import re
from enum import Enum
from os import getenv
from typing import Optional, Union

import attrs
import cattrs
import numpy as np
from dotenv import load_dotenv

from streetdownloader.common import BaseImage, Location, LimitedClientSession, image_from_res

GOOGLE_SV_API = 'https://maps.googleapis.com/maps/api/streetview'
MAX_API_REQ_PER_MIN = 30000
CROP_GOOGLE_LOGO = False

load_dotenv()
GOOGLE_API_KEY = getenv("GOOGLE_API_KEY")
GOOGLE_API_SIGNATURE = getenv("GOOGLE_API_SIGNATURE")


class MetadataStatus(Enum):
    """
    OK: Indicates that no errors occurred; a panorama is found and metadata is returned.
    ZERO_RESULTS: Indicates that no panorama could be found near the provided location.
        This may occur if a non-existent or invalid panorama ID is given.
    NOT_FOUND: Indicates that the address string provided in the location parameter could not be found.
        This may occur if a non-existent address is given.
    OVER_QUERY_LIMIT: Indicates that you have exceeded your daily quota or per-second quota for this API.
    REQUEST_DENIED: Indicates that your request was denied. This may occur if you did not authorize your request,
        or if the Street View Static API is not activated in the Google Cloud Console project containing your API key.
    INVALID_REQUEST: Generally indicates that the query parameters (address or latlng or components) are missing.
    UNKNOWN_ERROR: Indicates that the request could not be processed due to a server error.
        This is often a temporary status. The request may succeed if you try again.
    """
    OK = 'OK'
    ZERO_RESULTS = 'ZERO_RESULTS'
    NOT_FOUND = 'NOT_FOUND'
    OVER_QUERY_LIMIT = 'OVER_QUERY_LIMIT'
    REQUEST_DENIED = 'REQUEST_DENIED'
    INVALID_REQUEST = 'INVALID_REQUEST'
    UNKNOWN_ERROR = 'UNKNOWN_ERROR'


cattrs.register_unstructure_hook(MetadataStatus, lambda x: x.value)


@attrs.define
class Metadata:
    status: MetadataStatus
    copyright: str = None
    date: str = None
    location: Location = None
    pano_id: str = None

    def __bool__(self):
        return self.ok

    @property
    def ok(self):
        return self.status == MetadataStatus.OK

    @property
    def from_google(self):
        return self.copyright == '© Google'


@attrs.define
class RequestParams:
    pano: str = None
    heading: float = None  # 0 < heading < 360
    fov: float = 90  # 0 < fov < 120
    pitch: float = 0  # -90 < pitch < 90


@attrs.define
class View(BaseImage):
    metadata: Metadata = None
    params: RequestParams = None
    api_image: np.ndarray = None
    crop_logo: bool = False

    @property
    def location(self) -> Location:
        return self.metadata.location

    @property
    def image(self):
        if self.api_image is None:
            return
        if self.crop_logo:
            height, width = self.api_image.shape[:2]
            return self.api_image[0:height - 22, 0:width]
        return self.api_image

    @property
    def default_filename(self):
        return "%s_%03d_%02d_%03d.jpg" % (
            self.metadata.pano_id or self.params.pano, self.params.heading, self.params.fov, self.params.pitch + 90)


def _prompt_google_api_key():
    from getpass import getpass
    global GOOGLE_API_KEY
    print("Missing Google API key. Insert GOOGLE_API_KEY in a '.env' file.")
    print("Get your API key at: https://developers.google.com/maps/documentation/streetview/get-api-key")
    api_key = getpass('Google API key:').strip()
    assert re.match("^AIzaSy[A-Za-z0-9_-]{33}$", api_key), "Invalid API key"
    GOOGLE_API_KEY = api_key


class _SessionManager:
    _session: Optional[LimitedClientSession] = None
    _active = 0

    @classmethod
    def get(cls):
        if cls._active == 0:
            cls._session = LimitedClientSession(MAX_API_REQ_PER_MIN, 60)
        cls._active += 1
        return cls._session

    @classmethod
    async def close(cls):
        cls._active -= 1
        if cls._active == 0:
            await cls._session.close()


class StreetViewAPI:
    def __init__(self):
        self._session = _SessionManager.get()
        self._closed = False

    def __enter__(self) -> None:
        raise TypeError("Use async with instead")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.close()

    async def close(self):
        if not self._closed:
            await _SessionManager.close()
            self._closed = True

    def _api_get(self, url: str, params: dict):
        if GOOGLE_API_KEY is None:
            _prompt_google_api_key()
        params = {**params, 'size': '640x640', 'key': GOOGLE_API_KEY}
        if GOOGLE_API_SIGNATURE:
            params['signature'] = GOOGLE_API_SIGNATURE
        return self._session.get(url, params=params)

    async def request_metadata(self, location: Union[Location, str, None] = None,
                               pano: Union[str, None] = None) -> Metadata:
        params = {}
        if pano:
            params['pano'] = pano
        elif location:
            if isinstance(location, tuple):
                location = f'{location[0]},{location[1]}'
            params['location'] = location
        while not self._session.closed:
            request = await self._api_get(GOOGLE_SV_API + '/metadata', params)
            async with request as res:
                meta: Metadata = cattrs.structure(await res.json(), Metadata)
            if meta.status == MetadataStatus.OVER_QUERY_LIMIT:
                await asyncio.sleep(1)
                continue
            return meta

    async def request_image(self, view: View) -> Optional[View]:
        if view.metadata is None:
            view.metadata = await self.request_metadata(view.params.pano)
            view.params.pano = view.metadata.pano_id
        params = {k: v for k, v in attrs.asdict(view.params).items() if v is not None}
        request = await self._api_get(GOOGLE_SV_API, params)
        async with request as res:
            if res.ok:
                view.api_image = await image_from_res(res)
                return view
