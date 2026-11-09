"""Geometry helpers: straightening near-straight sides, edge lists."""
import math
from shapely.geometry import Polygon, LineString, Point

SQFT = 10.763910417  # square feet per square metre (international foot)


def _ring_simplify(coords, tol):
    pts = list(coords)
    if pts and pts[0] == pts[-1]:
        pts = pts[:-1]
    removed = []
    while len(pts) > 3:
        best, best_d = None, None
        n = len(pts)
        for i in range(n):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
            d = LineString([a, c]).distance(Point(b))
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best_d is None or best_d > tol:
            break
        removed.append(round(best_d, 3))
        pts.pop(best)
    return pts, removed


def straighten(poly, tol):
    """Remove corners lying within `tol` metres of the line joining their neighbours.
    Returns (new_polygon, list_of_removed_offsets)."""
    ext, removed = _ring_simplify(poly.exterior.coords, tol)
    holes = []
    for h in poly.interiors:
        hp, r = _ring_simplify(h.coords, tol)
        holes.append(hp)
        removed += r
    new = Polygon(ext, holes)
    if not new.is_valid:
        return poly, []
    return new, removed


def explode_polys(g):
    if g.geom_type == "Polygon":
        return [g]
    if g.geom_type == "MultiPolygon":
        return list(g.geoms)
    return []


def ring_edges(coords):
    c = list(coords)
    for i in range(len(c) - 1):
        yield c[i], c[i + 1], math.dist(c[i], c[i + 1])
