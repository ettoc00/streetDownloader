import asyncio
import itertools
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

from streetdownloader.common import Location, sync_generator, LimitedClientSession, DEFAULT_POVS_ARGS, DEFAULT_FOV
from streetdownloader.location import grid_coords_between
from streetdownloader.panoscraper import scrape_panorama, Panorama, DEFALUT_ZOOM_LEVEL
from streetdownloader.streetviewapi import StreetViewAPI, Metadata, View, RequestParams

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


async def generate_metadata_from_coords(coords: Iterable[Location], from_google: bool = False,
                                        also_errors: bool = False):
    panos = set()
    iterator, chunk_size = iter(coords), 1 << 16
    async with StreetViewAPI() as api:
        while True:
            chunk = tuple(islice(iterator, chunk_size))
            if not chunk:
                return
            for coro in asyncio.as_completed(map(api.request_metadata, chunk)):
                metadata: Metadata = await coro
                if metadata.ok:
                    if (not from_google or metadata.from_google) and metadata.pano_id not in panos:
                        panos.add(metadata.pano_id)
                        yield metadata
                    elif also_errors:
                        yield
                elif also_errors:
                    yield metadata


async def generate_metadata(loc1: Location, loc2: Location, from_google: bool = False, empty_if_existing: bool = False):
    loc1, loc2 = Location.of(loc1), Location.of(loc2)
    generator, _ = grid_coords_between(loc1, loc2)
    async for metadata in generate_metadata_from_coords(generator, from_google, empty_if_existing):
        yield metadata


def generate_povs(n: int, pitch: int, fov: int = DEFAULT_FOV) -> Iterable[tuple[int, int, int]]:
    return ((int(i * 360.0 / n), int(fov), int(pitch)) for i in range(n))


def generate_all_povs(*povs_args):
    return itertools.chain.from_iterable(generate_povs(*args) for args in povs_args)


async def retrieve_panoramas_from_metadata(metadata: Iterable[Metadata], folder: Optional['StrOrBytesPath'] = None,
                                           zoom: int = DEFALUT_ZOOM_LEVEL, skip_existing=False,
                                           empty_if_existing: bool = False):
    if skip_existing or empty_if_existing:
        assert folder is not None
        assert not (skip_existing and empty_if_existing)
        folder = Path(folder)
    async with LimitedClientSession(500, 10) as limited_session:
        for m in metadata:
            panorama = Panorama(m)
            if skip_existing or empty_if_existing:
                if (folder / panorama.default_filename).exists():
                    if empty_if_existing:
                        yield panorama
                    continue
            await scrape_panorama(limited_session, panorama, zoom)
            yield panorama


async def retrieve_panoramas(loc1: Location, loc2: Location, folder: Optional['StrOrBytesPath'] = None,
                             zoom: int = DEFALUT_ZOOM_LEVEL, skip_existing=False,
                             empty_if_existing: bool = False):
    metadata_generator = sync_generator(generate_metadata(loc1, loc2), asyncio.get_running_loop())
    async for panorama in retrieve_panoramas_from_metadata(
            metadata_generator, folder, zoom, skip_existing, empty_if_existing):
        yield panorama


def download_panoramas(folder: 'StrOrBytesPath', loc1: Location, loc2: Location, skip_existing: bool = True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for panorama in sync_generator(retrieve_panoramas(loc1, loc2, folder, skip_existing)):
        panorama.save(folder)


async def retrieve_views_from_metadata(metadata: Iterable[Metadata], folder: Optional['StrOrBytesPath'] = None,
                                       povs_args=DEFAULT_POVS_ARGS, skip_existing: bool = False,
                                       empty_if_existing: bool = False):
    if skip_existing or empty_if_existing:
        assert folder is not None
        assert not (skip_existing and empty_if_existing)
        folder = Path(folder)
    async with StreetViewAPI() as api:
        for m in metadata:
            coros = []
            for heading, fov, pitch in generate_all_povs(*povs_args):
                view = View(m, RequestParams(m.pano_id, heading, fov, pitch))
                if skip_existing or empty_if_existing:
                    view_img = view.default_filename
                    if (folder / view_img).exists():
                        if empty_if_existing:
                            yield view
                        continue
                coros.append(api.request_image(view))
            for coro in asyncio.as_completed(coros):
                view: Optional[View] = await coro
                if view:
                    yield view


async def retrieve_views(loc1: Location, loc2: Location, folder: Optional['StrOrBytesPath'] = None,
                         povs_args=DEFAULT_POVS_ARGS, skip_existing: bool = False, empty_if_existing: bool = False):
    metadata_generator = sync_generator(generate_metadata(loc1, loc2), asyncio.get_running_loop())
    async for view in retrieve_views_from_metadata(
            metadata_generator, folder, povs_args, skip_existing, empty_if_existing):
        yield view


def download_views(folder: 'StrOrBytesPath', loc1: Location, loc2: Location, povs_args=DEFAULT_POVS_ARGS,
                   skip_existing: bool = True, link_in_folder=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if link_in_folder is not None:
        link_in_folder = Path(link_in_folder)
    for view in sync_generator(retrieve_views(loc1, loc2, folder, povs_args, skip_existing)):
        path = view.save(folder, False, True)
        if link_in_folder:
            (link_in_folder / path.name).symlink_to(path)
