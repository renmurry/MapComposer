"""Vector input: SHP / GPKG / KML / KMZ / GeoJSON / CSV points -> GeoDataFrames."""
import os
import re
import warnings

import pandas as pd
import geopandas as gpd
import pyogrio
from shapely import force_2d, make_valid

VECTOR_EXT = (".shp", ".gpkg", ".kml", ".kmz", ".geojson", ".json", ".csv")

_LAT = re.compile(r"^(lat|latitude|y_lat|lat_dd|lat_deg)$", re.I)
_LON = re.compile(r"^(lon|long|longitude|lng|x_lon|lon_dd|long_dd|lon_deg)$", re.I)
_E = re.compile(r"^(e|east|easting|x|utm_e|x_utm|eastings)$", re.I)
_N = re.compile(r"^(n|north|northing|y|utm_n|y_utm|northings)$", re.I)


class VLayer:
    """One vector layer. `gdf` stays in its source CRS; `to_map(crs)` gives a reprojected copy."""

    def __init__(self, path, layer=None, gdf=None, note=None):
        self.path = path
        self.sublayer = layer
        base = os.path.splitext(os.path.basename(path))[0]
        self.name = f"{base}:{layer}" if layer and layer != base else base
        self.note = note
        if gdf is None:
            gdf = pyogrio.read_dataframe(path, layer=layer)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        if len(gdf):
            gdf["geometry"] = force_2d(gdf.geometry.values)
            bad = ~gdf.geometry.is_valid
            if bad.any():
                gdf.loc[bad, "geometry"] = make_valid(gdf.geometry[bad].values)
        self.gdf = gdf.reset_index(drop=True)
        self.crs = gdf.crs
        kinds = {t.replace("Multi", "") for t in self.gdf.geom_type.unique()} - {"GeometryCollection"}
        self.kind = kinds.pop() if len(kinds) == 1 else ("Mixed" if kinds else "Empty")
        self.fields = [c for c in self.gdf.columns if c != "geometry"]
        self._map = {}

    def __len__(self):
        return len(self.gdf)

    def to_map(self, crs):
        key = str(crs)
        if key not in self._map:
            if self.crs is None:
                g = self.gdf.set_crs(crs, allow_override=True)
            else:
                g = self.gdf.to_crs(crs)
            self._map[key] = g
        return self._map[key]

    def text_values(self, field):
        return [str(v) for v in self.gdf[field].tolist() if v is not None and str(v) not in ("", "nan", "None")]

    def string_fields(self, max_unique=40):
        out = []
        for f in self.fields:
            s = self.gdf[f]
            if pd.api.types.is_string_dtype(s) or s.dtype == object or pd.api.types.is_integer_dtype(s):
                u = s.dropna().astype(str).unique()
                if 1 <= len(u) <= max_unique:
                    out.append(f)
        return out

    def label_fields(self):
        return [f for f in self.fields if self.gdf[f].notna().all() and len(self.gdf)]


def _read_csv(path, map_crs=None):
    df = pd.read_csv(path)
    cols = list(df.columns)
    lat = next((c for c in cols if _LAT.match(str(c).strip())), None)
    lon = next((c for c in cols if _LON.match(str(c).strip())), None)
    if lat and lon:
        df = df.dropna(subset=[lat, lon])
        g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon].astype(float), df[lat].astype(float)), crs="EPSG:4326")
        return g, f"points from columns {lon}/{lat} (WGS 84)"
    e = next((c for c in cols if _E.match(str(c).strip())), None)
    n = next((c for c in cols if _N.match(str(c).strip())), None)
    if e and n:
        df = df.dropna(subset=[e, n])
        g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[e].astype(float), df[n].astype(float)), crs=None)
        return g, f"points from columns {e}/{n} - assumed to be in the map's coordinate system"
    raise ValueError("CSV has no lat/long or easting/northing columns")


def read_any(path):
    """Return a list of VLayer for one file (GPKG/KML can hold several layers)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        gdf, note = _read_csv(path)
        return [VLayer(path, gdf=gdf, note=note)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            subs = [l[0] for l in pyogrio.list_layers(path)]
        except Exception:
            subs = [None]
        out = []
        for s in subs if len(subs) > 1 else [None]:
            try:
                lyr = VLayer(path, layer=s)
                if len(lyr):
                    out.append(lyr)
            except Exception:
                continue
    return out


def find_vectors(folder):
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in sorted(files):
            if f.lower().endswith(VECTOR_EXT):
                # skip sidecar CSVs that describe raster classes
                if f.lower().endswith(".csv") and os.path.exists(os.path.join(root, f[:-4] + ".tif")):
                    continue
                out.append(os.path.join(root, f))
    return out
