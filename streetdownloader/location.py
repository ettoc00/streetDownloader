from math import ceil

import numpy as np
from geopy.distance import distance
from tqdm import tqdm

from streetdownloader.common import Location

DEFAULT_METERS_BETWEEN = 8


def grid_coords_between(loc1: Location, loc2: Location, meters_between: float = DEFAULT_METERS_BETWEEN):
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

    return (Location(lat, lng) for lat in lat_line for lng in lng_line), len(lat_line) * len(lng_line)


def locations_graph(*locs: Location):
    import networkx as nx
    from sklearn.neighbors import KDTree
    G = nx.Graph()
    G.add_nodes_from(locs)
    nx.set_node_attributes(G, {node: node for node in G.nodes()}, "pos")
    coords = np.array([list(l0) for l0 in locs])
    _, kd_indices = KDTree(coords).query(coords, k=2)
    indices = set(map(tuple, map(sorted, kd_indices)))
    for i, j in tqdm(indices):
        G.add_edge(locs[i], locs[j], length=distance(locs[i], locs[j]).m)
    return G
