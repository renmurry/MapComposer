"""Context detection: what is in the folder, what each layer is, which template fits."""
import os
import re
import warnings

from .io_vector import read_any, find_vectors
from .io_raster import RLayer, find_rasters
from .styles import load_styles, is_lulc_values

VROLES = ["Boundary", "Admin", "Thematic", "Road", "Drainage", "Building", "Points", "Contour", "Other", "Ignore"]
RROLES = ["Basemap", "Classified", "Elevation", "Continuous", "Ignore"]
KIND_TO_RROLE = {"rgb": "Basemap", "multiband": "Basemap", "classified": "Classified",
                 "elevation": "Elevation", "continuous": "Continuous"}

_NAME_RULES = [
    ("Boundary", r"bound|bndry|\bbnd\b|border|plot|parcel|\bsite\b|\baoi\b|study.?area|extent"),
    ("Thematic", r"lulc|land.?use|land.?cover|\blu\b|\blc\b|classif"),
    ("Admin", r"ward|vill|district|\bdist\b|block|circle|admin|mouza|constituen|town|municipal|colony|locality|\bkma\b|\bkmc\b|state|tehsil|taluk"),
    ("Road", r"road|(^|[_\W\d])rds?($|[_\W\d])|street|path|lane|highway|track|\bnh\d*\b"),
    ("Drainage", r"drain|stream|river|nala|nallah|canal|water.?way|channel|culvert|flow"),
    ("Building", r"build|bldg|footprint|house|struct|roof"),
    ("Contour", r"contour|cntr"),
    ("Points", r"point|\bpts\b|gcp|survey|poi|location|sample|landmark"),
]


def _first_field(layer, names):
    low = {f.lower(): f for f in layer.fields}
    for n in names:
        if n in low:
            return low[n]
    return ""


def guess_vector(layer):
    """Returns dict(role, style_field, label_field)."""
    st = load_styles()
    cf = st["class_fields"]
    n = layer.name.lower()
    kind = layer.kind
    role = None
    for r, pat in _NAME_RULES:
        if re.search(pat, n):
            if r in ("Boundary", "Admin", "Thematic", "Building") and kind != "Polygon":
                continue
            if r in ("Road", "Drainage", "Contour") and kind not in ("LineString", "Polygon"):
                continue
            role = r
            break
    sfields = layer.string_fields()
    # field semantics win over a missing/weak name match
    lulc_field = ""
    if kind == "Polygon":
        for f in [x for x in sfields if x.lower() in cf["lulc"]] + sfields:
            if is_lulc_values(layer.text_values(f)):
                lulc_field = f; break
    if lulc_field and role in (None, "Thematic", "Other"):
        role = "Thematic"
    if role is None:
        if kind == "Point":
            role = "Points"
        elif kind == "Polygon":
            nameish = _first_field(layer, cf["admin"])
            role = "Admin" if (len(layer) >= 3 and nameish) else ("Boundary" if len(layer) <= 10 else "Other")
        elif kind == "LineString":
            role = "Drainage" if _first_field(layer, cf["drainage"]) else "Other"
        else:
            role = "Other" if len(layer) else "Ignore"

    style_field = ""
    if role == "Thematic":
        style_field = lulc_field or (sfields[0] if sfields else "")
    elif role == "Road":
        style_field = next((f for f in [_first_field(layer, [c]) for c in cf["road"]] if f and f in sfields), "")
        if not style_field:
            style_field = next((f for f in sfields if not any(ch.isdigit() for v in layer.text_values(f) for ch in v)), "")
    elif role == "Drainage":
        style_field = _first_field(layer, cf["drainage"])
    elif role in ("Building", "Other", "Points") and sfields:
        small = [f for f in sfields if 2 <= len(set(layer.text_values(f))) <= 8 and len(set(layer.text_values(f))) < len(layer)]
        style_field = small[0] if small else ""

    label_field = ""
    if role in ("Admin", "Points") or (role == "Boundary" and len(layer) > 1):
        cand = [_first_field(layer, [c]) for c in cf["admin"]] + layer.label_fields()
        for f in cand:
            if f and f in layer.fields:
                vals = layer.text_values(f)
                if len(vals) == len(layer) and len(set(vals)) >= max(1, int(0.8 * len(vals))):
                    label_field = f; break
    return dict(role=role, style_field=style_field, label_field=label_field)


def scan(folder, extra_files=()):
    """Scan a folder (recursively) and/or explicit files."""
    warns = []
    rpaths = find_rasters(folder) if folder else []
    vpaths = find_vectors(folder) if folder else []
    for f in extra_files:
        (rpaths if f.lower().endswith((".tif", ".tiff", ".img", ".jp2", ".vrt")) else vpaths).append(f)

    rasters = []
    for p in rpaths:
        try:
            rl = RLayer(p)
        except Exception as e:
            warns.append(f"Could not read raster {os.path.basename(p)}: {e}"); continue
        rasters.append(dict(layer=rl, role=KIND_TO_RROLE[rl.kind], bands=_default_bands(rl)))
    # keep one of each display role active by default (highest resolution basemap, DSM over DEM)
    for role in ("Basemap", "Elevation", "Classified", "Continuous"):
        same = [r for r in rasters if r["role"] == role]
        if len(same) > 1:
            same.sort(key=lambda r: (r["layer"].res_m, -r["layer"].width * r["layer"].height))
            for r in same[1:]:
                r["role"] = "Ignore"

    vectors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in vpaths:
            try:
                layers = read_any(p)
            except Exception as e:
                warns.append(f"Could not read {os.path.basename(p)}: {e}"); continue
            for lyr in layers:
                g = guess_vector(lyr)
                vectors.append(dict(layer=lyr, **g))
                if lyr.note:
                    warns.append(f"{lyr.name}: {lyr.note}")
    # prefer "<name>_simplified" boundaries over their originals
    names = {v["layer"].name.lower(): v for v in vectors}
    for v in vectors:
        n = v["layer"].name.lower()
        if n.endswith("_simplified") and n[:-11] in names and v["role"] == "Boundary":
            names[n[:-11]]["role"] = "Ignore"
    if not rasters and not vectors:
        warns.append("No rasters or vector files found")
    return dict(folder=folder, rasters=rasters, vectors=vectors, warnings=warns, name=default_name(folder))


def _default_bands(rl):
    b = rl.count - (1 if rl.has_alpha else 0)
    if rl.kind == "rgb" or (rl.kind == "multiband" and b == 3):
        return [1, 2, 3]
    if rl.kind == "multiband":
        return [1, 2, 3] if b >= 3 else [1, 1, 1]
    return [1]


def context(sc):
    roles = {}
    for v in sc["vectors"]:
        if v["role"] != "Ignore":
            roles[v["role"]] = roles.get(v["role"], 0) + len(v["layer"])
    kinds = {r["layer"].kind for r in sc["rasters"] if r["role"] != "Ignore"}
    lines = [v for v in sc["vectors"] if v["role"] != "Ignore"]
    lines_only = bool(lines) and all(v["layer"].kind == "LineString" for v in lines)
    return dict(roles=roles, raster_kinds=kinds, lines_only=lines_only)


def default_name(folder):
    if not folder:
        return "Map"
    name = os.path.basename(os.path.normpath(folder))
    name = re.sub(r"(?i)[\s_-]*processed[\s_-]*data", "", name).strip(" _-")
    return (name or "Map").replace("_", " ")
