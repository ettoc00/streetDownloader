from math import ceil

import numpy as np
from geopy.distance import distance

from streetdownloader.common import Location

METERS_BETWEEN = 8


def unique_coords_between(loc1: Location, loc2: Location, meters_between: int = METERS_BETWEEN) -> set[Location]:
    lat1, lat2 = sorted((loc1.lat, loc2.lat))
    lng1, lng2 = sorted((loc1.lng, loc2.lng))
    half_inv_mb = .5 / meters_between

    def eval_line(rev: bool):
        x1, x2 = (lng1, lng2) if rev else (lat1, lat2)
        if x1 == x2:
            return np.array([x1])
        q = (lat2, lng1), (lat1, lng2)
        d1 = distance((lat1, lng1), q[rev]).m
        d2 = distance((lat2, lng2), q[not rev]).m
        return np.linspace(x1, x2, ceil((d1 + d2) * half_inv_mb) + 1)

    lat_line = eval_line(False)
    lng_line = eval_line(True)

    grid = np.stack(np.meshgrid(lat_line, lng_line), -1)
    return set(map(Location.of, np.around(grid, 7).reshape(-1, 2)))
