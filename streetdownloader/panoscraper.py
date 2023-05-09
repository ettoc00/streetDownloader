import asyncio
import json

import attrs
import cv2
import numpy as np
from aiohttp import ClientSession

from streetdownloader.common import BaseImage, image_from_res, image_from_bytes, Location
from streetdownloader.streetviewapi import Metadata, MetadataStatus

DEFALUT_ZOOM_LEVEL = 4
GOOGLE_SV_TILE_API = 'https://streetviewpixels-pa.googleapis.com/v1/tile'
GOOGLE_SV_PHOTOMETA = 'https://www.google.com/maps/photometa/v1?authuser=0&hl=en&gl=nl&pb=!1m4!1smaps_sv.tactile!11m2' \
                      '!2m1!1b1!2m2!1sen!2snl!3m3!1m2!1e2!2s{}!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48' \
                      '!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!' \
                      '2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3'


@attrs.define
class Photometa:
    pano_id: str = None
    location: Location = None
    copyright: str = None
    near: list['Photometa'] = None

    def as_metadata(self):
        return Metadata(status=MetadataStatus.OK, copyright='© ' + self.copyright,
                        location=self.location, pano_id=self.pano_id)


@attrs.define
class Panorama(BaseImage):
    metadata: Metadata = None
    photometa: Photometa = None
    tiles: np.ndarray = None

    @property
    def image(self):
        if self.tiles is not None:
            return np.hstack(np.hstack(self.tiles))

    @property
    def pano_id(self) -> str:
        if self.metadata and self.metadata.pano_id:
            return self.metadata.pano_id
        if self.photometa and self.photometa.pano_id:
            return self.photometa.pano_id

    @property
    def location(self) -> Location:
        if self.photometa and self.photometa.location:
            return self.photometa.location
        if self.metadata and self.metadata.location:
            return self.metadata.location

    def load_tiles_from_image(self, image: np.ndarray):
        pass

    def load_tiles_from_path(self, path):
        with open(path, 'rb') as f:
            return self.load_tiles_from_image(image_from_bytes(f.read()))

    @property
    def default_filename(self):
        return "%s.jpg" % self.pano_id


async def get_panorama_tile(session: ClientSession, pano_id: str, zoom: int, coords: tuple[int, int]) -> np.ndarray:
    async with session.get(GOOGLE_SV_TILE_API, params={
        'cb_client': 'maps_sv.tactile', 'panoid': pano_id,
        'x': coords[0], 'y': coords[1],
        'zoom': zoom, 'nbt': 1, 'fover': 2
    }) as res:
        img = await image_from_res(res)
        return img


async def get_panorama_photometa(session: ClientSession, pano_id: str):
    def location_from_meta_list(q):
        return Location(q[2], q[3])

    async with session.get(GOOGLE_SV_PHOTOMETA.format(pano_id)) as res:
        if not res.ok:
            return None
        content = await res.text()
    code, _, json_data = content.partition('\n')
    data = json.loads(json_data)
    pm = Photometa(pano_id, location_from_meta_list(data[1][0][5][0][1][0]), data[1][0][4][1][0][0][0], [
        Photometa(k[0][1], location_from_meta_list(k[2][0])) for k in data[1][0][5][0][3][0]
    ])
    return pm


async def scrape_panorama(session: ClientSession, panorama: Panorama, zoom: int = DEFALUT_ZOOM_LEVEL):
    tiles_coords_list = tuple((i, j) for j in range(2 ** (zoom - 1)) for i in range(2 ** zoom))
    coros = [get_panorama_tile(session, panorama.pano_id, zoom, xy) for xy in tiles_coords_list]
    if panorama.photometa:
        tiles = await asyncio.gather(*coros)
    else:
        *tiles, pm = await asyncio.gather(*coros, get_panorama_photometa(session, panorama.pano_id))
        panorama.photometa = pm
    if all(tile is None for tile in tiles):
        return Panorama
    tile_shape = max(tile.shape for tile in tiles if tile is not None)
    reshaped = np.zeros((2 ** (zoom - 1), 2 ** zoom, *tile_shape))
    tiles_with_coords = tuple(zip(tiles, tiles_coords_list))
    for tile, coords in tiles_with_coords:
        if tile is not None:
            i, j = coords
            if tile.shape == tile_shape:
                reshaped[j, i] = tile
            else:
                reshaped[j, i] = cv2.resize(tile, tile_shape[:2])
    mask = np.any(reshaped, axis=(2, 3, 4))
    nonzero_coords = np.argwhere(mask)
    if nonzero_coords.size > 0:
        min_y, min_x = nonzero_coords.min(axis=0)[:2]
        max_y, max_x = nonzero_coords.max(axis=0)[:2]
        panorama.tiles = reshaped[min_y:max_y + 1, min_x:max_x + 1]
    return panorama
