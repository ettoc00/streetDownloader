import asyncio

import attrs
import numpy as np
from aiohttp import ClientSession

from streetdownloader.common import BaseImage, image_from_res

ZOOM_LEVEL = 4
GOOGLE_SV_TILE_API = 'https://streetviewpixels-pa.googleapis.com/v1/tile?&x=11&y=3&zoom=4&'


@attrs.define
class Panorama(BaseImage):
    pano_id: str
    tiles: np.ndarray = None

    @property
    def image(self):
        if self.tiles is not None:
            return np.hstack(np.hstack(self.tiles))


async def get_panorama_tile(session: ClientSession, pano_id: str, zoom: int, coords: tuple[int, int]) -> np.ndarray:
    async with session.get(GOOGLE_SV_TILE_API, params={
        'cb_client': 'maps_sv.tactile', 'panoid': pano_id,
        'x': coords[0], 'y': coords[1],
        'zoom': zoom, 'nbt': 1, 'fover': 2
    }) as res:
        return await image_from_res(res)


async def scrape_panorama(session: ClientSession, pano_id: str, zoom: int = ZOOM_LEVEL):
    panorama = Panorama(pano_id)
    tiles_coords_list = tuple((i, j) for j in range(2 ** (zoom - 1)) for i in range(2 ** zoom))
    tiles = await asyncio.gather(*map(lambda xy: get_panorama_tile(session, pano_id, zoom, xy), tiles_coords_list))
    if all(tile is not None for tile in tiles):
        panorama.tiles = np.array(tiles).reshape((8, 16, 512, 512, 3))
    return panorama
