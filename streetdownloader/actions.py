import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm.asyncio import tqdm

from streetdownloader.common import Location, sync_generator, LimitedClientSession
from streetdownloader.location import unique_coords_between
from streetdownloader.panoscraper import scrape_panorama, Panorama
from streetdownloader.streetviewapi import StreetViewAPI, Metadata, View

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath

PANORAMA_FORMAT = "%s.jpg"
VIEW_FORMAT = "%s_%03d_%02d_%03d.jpg"


async def generate_metadata(api: StreetViewAPI, loc1: Location, loc2: Location):
    panos = set()
    loc1, loc2 = Location.of(loc1), Location.of(loc2)
    tasks = tuple(map(api.request_metadata, unique_coords_between(loc1, loc2)))
    for coro in tqdm.as_completed(tasks):
        metadata: Metadata = await coro
        if metadata.status == 'OK' and metadata.copyright == '© Google' and metadata.pano_id not in panos:
            panos.add(metadata.pano_id)
            yield metadata


async def retrieve_panoramas(loc1: Location, loc2: Location,
                             folder: 'StrOrBytesPath | None' = None, skip_existing=False):
    if skip_existing:
        assert folder is not None
        folder = Path(folder)
    async with StreetViewAPI() as api, LimitedClientSession(50, 10) as limited_session:
        async for metadata in generate_metadata(api, loc1, loc2):
            if skip_existing:
                if (folder / (PANORAMA_FORMAT % metadata.pano_id)).exists():
                    continue
            panorama: Panorama = await scrape_panorama(limited_session, metadata.pano_id)
            if panorama.tiles is not None:
                yield panorama


def download_panoramas(folder: 'StrOrBytesPath', loc1: Location, loc2: Location, skip_existing: bool = True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for panorama in sync_generator(retrieve_panoramas(loc1, loc2, folder, skip_existing)):
        panorama.save(folder / (PANORAMA_FORMAT % panorama.pano_id))


async def retrieve_views(loc1: Location, loc2: Location,
                         folder: 'StrOrBytesPath | None' = None, skip_existing=False):
    if skip_existing:
        assert folder is not None
        folder = Path(folder)
    rps = tuple([(int(i * 360 / 8), 90, 0) for i in range(8)] + [(int(i * 360 / 6), 90, 45) for i in range(6)])

    async with StreetViewAPI() as api:
        async for metadata in generate_metadata(api, loc1, loc2):
            coros = []
            for heading, fov, pitch in rps:
                if skip_existing:
                    view_img = VIEW_FORMAT % (metadata.pano_id, heading, fov, pitch + 90)
                    if (folder / view_img).exists():
                        continue
                coros.append(api.request_image(None, metadata, heading, fov, pitch))
            for coro in asyncio.as_completed(coros):
                view: View | None = await coro
                if view:
                    yield view


def download_views(folder: 'StrOrBytesPath', loc1: Location, loc2: Location, skip_existing: bool = True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for view in sync_generator(retrieve_views(loc1, loc2, folder, skip_existing)):
        name = VIEW_FORMAT % (view.metadata.pano_id, view.params.heading, view.params.fov, view.params.pitch + 90)
        view.save(folder / name)
