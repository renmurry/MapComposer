"""Map templates (YAML). User copies in %APPDATA%/MapComposer/templates override built-ins with the same id."""
import os
import copy
import yaml

from .paths import resource, user_dir

DEFAULTS = {
    "page": {"paper": "A3", "orientation": "landscape"},
    "text": {"title": "{name}", "subtitle": ""},
    "extent": {"roles": [], "rasters": [], "margin_mm": 18},
    "grid": "utm",
    "basemap": True,
    "elevation": {"hillshade": True, "opacity": 1.0},
    "boundary": {"side_lengths": False, "corners": False, "area_label": False},
    "labels": {"Admin": True, "Points": True},
    "stats": "none",
    "panel": ["title", "locator", "legend", "stats", "scale", "notes"],
    "panel_portrait": [["title", "scale", "notes"], ["legend", "stats"], ["locator"]],
    "notes": ["crs", "imagery", "source", "measurements", "north_date", "disclaimer"],
    "contours": {"default": False, "interval": 2},
    "suggest": [],
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_templates():
    found = {}
    for d in (resource("templates"), os.path.join(user_dir(), "templates")):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith((".yaml", ".yml")):
                continue
            p = os.path.join(d, f)
            try:
                with open(p, encoding="utf-8") as fh:
                    t = yaml.safe_load(fh) or {}
                t = _merge(DEFAULTS, t)
                t.setdefault("id", os.path.splitext(f)[0])
                t.setdefault("name", t["id"])
                t["_path"] = p
                found[t["id"]] = t
            except Exception as e:
                found[f"error:{f}"] = {"id": f"error:{f}", "name": f"(broken template {f}: {e})", "_broken": True}
    return [t for t in found.values() if not t.get("_broken")]


def _rule_points(rule, ctx):
    pts = rule.get("points", 0)
    if "has_role" in rule:
        return pts if ctx["roles"].get(rule["has_role"], 0) > 0 else 0
    if "has_raster" in rule:
        return pts if rule["has_raster"] in ctx["raster_kinds"] else 0
    if "no_raster" in rule:
        return pts if rule["no_raster"] not in ctx["raster_kinds"] else 0
    if "max_features" in rule:
        m = rule["max_features"]; n = ctx["roles"].get(m["role"], 0)
        return pts if 0 < n <= m["n"] else 0
    if "min_features" in rule:
        m = rule["min_features"]; n = ctx["roles"].get(m["role"], 0)
        return pts if n >= m["n"] else 0
    if "lines_only" in rule:
        return pts if ctx["lines_only"] else 0
    return 0


def suggest(templates, ctx):
    """ctx: dict(roles={role: feature_count}, raster_kinds=set, lines_only=bool). Returns [(score, template)]."""
    ranked = [(sum(_rule_points(r, ctx) for r in t.get("suggest", [])), t) for t in templates]
    ranked.sort(key=lambda x: -x[0])
    return ranked
