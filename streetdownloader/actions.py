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


async def generate_metadata(api: StreetViewAPI, loc1: Location, loc2: Location):
    panos = set()
    loop = asyncio.get_event_loop()
    loc1, loc2 = Location.of(loc1), Location.of(loc2)
    tasks = tuple(map(loop.create_task, map(api.request_metadata, unique_coords_between(loc1, loc2))))
    for coro in tqdm.as_completed(tasks):
        metadata: Metadata = await coro
        if metadata.status == 'OK' and metadata.copyright == '© Google' and metadata.pano_id not in panos:
            panos.add(metadata.pano_id)
            yield metadata


async def retrieve_panoramas(loc1: Location, loc2: Location):
    loop = asyncio.get_event_loop()
    tasks = []
    async with StreetViewAPI() as api, LimitedClientSession(50, 20) as limited_session:
        async for metadata in generate_metadata(api, loc1, loc2):
            tasks.append(loop.create_task(scrape_panorama(limited_session, metadata.pano_id)))
        for coro in tqdm.as_completed(tasks):
            panorama: Panorama = await coro
            if panorama.tiles is not None:
                yield panorama
    await api.close()


def download_panoramas(folder: 'StrOrBytesPath', loc1: Location, loc2: Location):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for panorama in sync_generator(retrieve_panoramas(loc1, loc2)):
        path = folder / (panorama.pano_id + '.jpg')
        panorama.save(path)


async def retrieve_views(loc1: Location, loc2: Location):
    loop = asyncio.get_event_loop()
    rps = tuple([(int(i * 360 / 8), 90, 0) for i in range(8)] + [(int(i * 360 / 6), 90, 45) for i in range(6)])
    tasks = []
    async with StreetViewAPI() as api:
        async for metadata in generate_metadata(api, loc1, loc2):
            for heading, fov, pitch in rps:
                tasks.append(loop.create_task(api.request_image(None, metadata, heading, fov, pitch)))
        for coro in tqdm.as_completed(tasks):
            view: View = await coro
            if view:
                yield view


def download_views(folder: 'StrOrBytesPath', loc1: Location, loc2: Location):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for view in sync_generator(retrieve_views(loc1, loc2)):
        name = f"{view.metadata.pano_id}_{view.params.heading:03}_{view.params.fov:2}_{(view.params.pitch + 90):03}.jpg"
        view.save(folder / name)
