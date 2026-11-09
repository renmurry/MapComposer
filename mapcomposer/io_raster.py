"""Raster input: kind detection and display/statistics reads, reprojected on the fly to the map CRS."""
import math
import os
import csv

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds, Window, bounds as win_bounds
from pyproj import CRS

RASTER_EXT = (".tif", ".tiff", ".img", ".jp2", ".vrt")


class RLayer:
    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        with rasterio.open(path) as r:
            self.count, self.dtype = r.count, r.dtypes[0]
            self.width, self.height = r.width, r.height
            self.crs = CRS.from_wkt(r.crs.to_wkt()) if r.crs else None
            self.res = abs(r.transform.a)
            self.bounds = tuple(r.bounds)
            self.nodata = r.nodata
            self.colorinterp = [c.name for c in r.colorinterp]
            try:
                self.colormap = r.colormap(1) if self.count == 1 else None
            except ValueError:
                self.colormap = None
            self.has_alpha = "alpha" in self.colorinterp
            self.kind = self._detect(r)
        self.classes = self._load_classes() if self.kind == "classified" else None

    @property
    def res_m(self):
        if self.crs is not None and self.crs.is_geographic:
            return self.res * 111320.0
        return self.res

    def _detect(self, r):
        n = self.name.lower()
        bands = self.count - (1 if self.has_alpha else 0)
        if bands >= 3:
            return "rgb" if (self.dtype == "uint8" and self.res_m < 1.0 and bands <= 3) else "multiband"
        if bands == 2:
            return "multiband"
        # single band
        if self.colormap is not None and self.dtype in ("uint8", "uint16", "int16", "int32", "uint32"):
            return "classified"
        if any(k in n for k in ("dsm", "dem", "dtm", "elev", "srtm", "alos", "height")):
            return "elevation"
        f = max(1, int(max(r.width, r.height) / 400))
        a = r.read(1, out_shape=(max(1, r.height // f), max(1, r.width // f)), resampling=Resampling.nearest,
                   masked=True)
        vals = a.compressed()
        if vals.size == 0:
            return "continuous"
        if np.issubdtype(vals.dtype, np.integer) or np.all(np.mod(vals, 1) == 0):
            u = np.unique(vals)
            if len(u) <= 40:
                return "classified"
        if any(k in n for k in ("lulc", "class", "landuse", "landcover", "lc_")):
            return "classified"
        return "continuous"

    def _load_classes(self):
        """value -> dict(name, color). From sidecar CSV (value,name[,color]) and/or embedded colour table."""
        classes = {}
        side = os.path.splitext(self.path)[0] + ".csv"
        if os.path.exists(side):
            with open(side, newline="", encoding="utf-8-sig") as f:
                for row in csv.reader(f):
                    if not row or not row[0].strip().lstrip("-").isdigit():
                        continue
                    v = int(row[0]); classes[v] = dict(name=row[1].strip() if len(row) > 1 else f"Class {v}",
                                                       color=row[2].strip() if len(row) > 2 and row[2].strip() else None)
        if self.colormap:
            for v, c in self.colormap.items():
                if v in classes and not classes[v]["color"]:
                    classes[v]["color"] = "#%02x%02x%02x" % tuple(c[:3])
                elif v not in classes and c[3] > 0 and c[:3] != (0, 0, 0):
                    classes[v] = dict(name=f"Class {v}", color="#%02x%02x%02x" % tuple(c[:3]))
        return classes

    # ---------------- reads ----------------
    def _open(self, map_crs):
        src = rasterio.open(self.path)
        if self.crs is not None and map_crs is not None and self.crs != map_crs:
            rs = Resampling.nearest if self.kind == "classified" else Resampling.bilinear
            return src, WarpedVRT(src, crs=map_crs.to_wkt(), resampling=rs)
        return src, None

    def read(self, map_crs, ext, max_px, bands=None, resampling=None):
        """Read bands for ground extent ext=(minx,maxx,miny,maxy) in map CRS, clipped to raster.
        Returns (array[b,h,w], valid_mask[h,w], (l,r,b,t)) or (None, None, None)."""
        src, vrt = self._open(map_crs)
        ds = vrt or src
        try:
            full = Window(0, 0, ds.width, ds.height)
            w = from_bounds(ext[0], ext[2], ext[1], ext[3], ds.transform)
            try:
                w = w.intersection(full)
            except Exception:
                return None, None, None
            c0, r0 = int(math.floor(w.col_off)), int(math.floor(w.row_off))
            c1 = int(math.ceil(w.col_off + w.width)); r1 = int(math.ceil(w.row_off + w.height))
            w = Window(c0, r0, max(1, c1 - c0), max(1, r1 - r0))
            f = max(1.0, max(w.width, w.height) / max_px)
            oh, ow = max(1, int(round(w.height / f))), max(1, int(round(w.width / f)))
            if resampling is None:
                resampling = Resampling.nearest if self.kind == "classified" else (
                    Resampling.average if f > 1.5 else Resampling.bilinear)
            bands = bands or list(range(1, ds.count + 1))
            arr = ds.read(bands, window=w, out_shape=(len(bands), oh, ow), resampling=resampling, masked=True)
            mask = ~np.any(np.ma.getmaskarray(arr), axis=0)
            if self.has_alpha:
                alpha_idx = self.colorinterp.index("alpha") + 1
                a = ds.read(alpha_idx, window=w, out_shape=(oh, ow), resampling=Resampling.nearest)
                mask &= a > 0
            elif vrt is not None or self.nodata is None:
                dm = ds.dataset_mask(window=w, out_shape=(oh, ow), resampling=Resampling.nearest)
                mask &= dm > 0
            l, b, r, t = win_bounds(w, ds.transform)
            return np.ma.getdata(arr), mask, (l, r, b, t)
        finally:
            if vrt is not None:
                vrt.close()
            src.close()

    def class_counts(self, map_crs, aoi=None, max_cells=6_000_000):
        """Pixel counts per class (optionally inside an AOI polygon in map CRS) -> (dict value->area_m2)."""
        from rasterio.features import geometry_mask
        src, vrt = self._open(map_crs)
        ds = vrt or src
        try:
            if aoi is not None:
                minx, miny, maxx, maxy = aoi.bounds
                w = from_bounds(minx, miny, maxx, maxy, ds.transform).intersection(Window(0, 0, ds.width, ds.height))
            else:
                w = Window(0, 0, ds.width, ds.height)
            w = Window(int(w.col_off), int(w.row_off), max(1, int(math.ceil(w.width))), max(1, int(math.ceil(w.height))))
            f = max(1.0, math.sqrt(w.width * w.height / max_cells))
            oh, ow = max(1, int(w.height / f)), max(1, int(w.width / f))
            a = ds.read(1, window=w, out_shape=(oh, ow), resampling=Resampling.nearest, masked=True)
            valid = ~np.ma.getmaskarray(a)
            tr = ds.window_transform(w) * rasterio.Affine.scale(w.width / ow, w.height / oh)
            if aoi is not None:
                inside = geometry_mask([aoi], out_shape=(oh, ow), transform=tr, invert=True)
                valid &= inside
            px_area = abs(tr.a * tr.e)
            vals, cnt = np.unique(np.ma.getdata(a)[valid], return_counts=True)
            return {int(v): float(c) * px_area for v, c in zip(vals, cnt)}, f > 1.0
        finally:
            if vrt is not None:
                vrt.close()
            src.close()


def find_rasters(folder):
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in sorted(files):
            if f.lower().endswith(RASTER_EXT):
                out.append(os.path.join(root, f))
    return out
