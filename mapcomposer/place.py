"""Place awareness from a folder of reference boundaries (state / district / block / village / ward ...)."""
import os
import re
import warnings

import pandas as pd

from .io_vector import read_any, find_vectors

LEVELS = [  # coarse -> fine
    ("state", r"state|\bst\b|india"),
    ("district", r"district|\bdist"),
    ("subdivision", r"sub.?div|tehsil|taluk"),
    ("block", r"block|\brd.?block"),
    ("circle", r"circle"),
    ("town", r"town|city|municipal|\bkma\b|\bkmc\b|urban"),
    ("village", r"vill|gram|mouza"),
    ("ward", r"ward"),
]
_NAME_FIELDS = ["name", "dist_name", "district", "dtname", "state_name", "st_nm", "stname", "state", "block_name",
                "block", "circle", "circle_nam", "vill_name", "village", "village_na", "name_1", "ward_name", "ward",
                "ward_no", "town", "town_name", "label"]


class PlaceIndex:
    def __init__(self, folder):
        self.folder = folder
        self.levels = []   # list of (level, layer, name_field)
        self.warnings = []
        if not folder or not os.path.isdir(folder):
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in find_vectors(folder):
                if p.lower().endswith(".csv"):
                    continue
                try:
                    for lyr in read_any(p):
                        if lyr.kind != "Polygon":
                            continue
                        lvl = next((l for l, pat in LEVELS if re.search(pat, lyr.name.lower())), None)
                        if not lvl:
                            self.warnings.append(f"Reference layer '{lyr.name}': level not recognised from its name "
                                                 "(use e.g. district, block, village, ward in the file name)")
                            continue
                        nf = self._name_field(lyr)
                        self.levels.append((lvl, lyr, nf))
                except Exception as e:
                    self.warnings.append(f"Reference {os.path.basename(p)}: {e}")
        order = [l for l, _ in LEVELS]
        self.levels.sort(key=lambda x: order.index(x[0]))

    @staticmethod
    def _name_field(lyr):
        low = {f.lower(): f for f in lyr.fields}
        for n in _NAME_FIELDS:
            if n in low and lyr.gdf[low[n]].notna().all():
                return low[n]
        for f in lyr.fields:
            if pd.api.types.is_string_dtype(lyr.gdf[f]) or lyr.gdf[f].dtype == object:
                return f
        return lyr.fields[0] if lyr.fields else ""

    def __bool__(self):
        return bool(self.levels)

    def lookup(self, point, crs):
        """point: shapely Point in `crs`. Returns {level: name}."""
        out = {}
        import geopandas as gpd
        pt = gpd.GeoSeries([point], crs=crs)
        for lvl, lyr, nf in self.levels:
            g = lyr.to_map(crs)
            hit = g[g.contains(pt.iloc[0])]
            if len(hit) and nf:
                out[lvl] = str(hit.iloc[0][nf]).strip().title() if str(hit.iloc[0][nf]).isupper() else str(hit.iloc[0][nf]).strip()
        return out

    def locator_layers(self, point, crs):
        """Context for the locator. Returns (outer_gdf, highlight_gdf, outer_level, highlight_level):
        all units of the preferred level (district first), with the unit containing the point highlighted."""
        import geopandas as gpd
        pt = gpd.GeoSeries([point], crs=crs).iloc[0]
        hits = {}
        for lvl, lyr, nf in self.levels:
            g = lyr.to_map(crs)
            h = g[g.contains(pt)]
            if len(h):
                hits[lvl] = (g, h)
        if not hits:
            return None, None, None, None
        for lvl in ("district", "subdivision", "block", "circle", "town", "village", "ward", "state"):
            if lvl in hits:
                g, h = hits[lvl]
                return g, h, lvl, lvl
        return None, None, None, None


def place_line(places):
    order = ["ward", "village", "town", "circle", "block", "subdivision", "district", "state"]
    parts = []
    for k in order:
        v = places.get(k)
        if v and v not in parts:
            parts.append(v + (" District" if k == "district" and "district" not in v.lower() else ""))
    return ", ".join(parts)


def fill_text(template, values):
    """Replace {key} placeholders; unknown/empty ones vanish along with dangling separators."""
    def rep(m):
        return str(values.get(m.group(1), "") or "")
    s = re.sub(r"\{(\w+)\}", rep, template)
    s = re.sub(r"\s*([,—–-])\s*(?=[,—–-]|$)", "", s)
    s = re.sub(r"^\s*[,—–-]\s*", "", s)
    return re.sub(r"\s{2,}", " ", s).strip()
