"""Style library: loads styles.yaml (user copy first) and resolves class colours."""
import os
import re
import yaml

from .paths import resource, user_dir

_cache = {}


def load_styles():
    p = os.path.join(user_dir(), "styles.yaml")
    if not os.path.exists(p):
        p = resource("styles.yaml")
    mt = os.path.getmtime(p)
    if _cache.get("path") != p or _cache.get("mtime") != mt:
        with open(p, encoding="utf-8") as f:
            _cache.update(path=p, mtime=mt, data=yaml.safe_load(f))
    return _cache["data"]


def match_rule(rules, value):
    v = str(value).lower()
    for r in rules or []:
        try:
            if re.search(r["match"], v):
                return r
        except re.error:
            continue
    return None


def is_lulc_values(values, min_share=0.6):
    st = load_styles()
    vals = [v for v in values if str(v).strip()]
    if not vals:
        return False
    uniq = set(map(str, vals))
    hits = sum(1 for u in uniq if match_rule(st["lulc"], u))
    return hits / len(uniq) >= min_share and len(uniq) >= 2


def class_colors(kind, values):
    """kind: road | lulc | generic. Returns {value: dict(color, width, dash)}."""
    st = load_styles()
    pal = st["fallback_palette"]
    out, k = {}, 0
    rules = st.get(kind) if kind in ("road", "lulc") else None
    for v in values:
        r = match_rule(rules, v) if rules else None
        if r:
            out[v] = dict(color=r["color"], width=r.get("width"), dash=r.get("dash", False))
        else:
            out[v] = dict(color=pal[k % len(pal)], width=None, dash=False); k += 1
    return out


def pretty(s):
    s = str(s).replace("_", " ").strip()
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)


# ---------------------------------------------------------------- themes & fonts
_theme_cache = {}
_fonts_loaded = False


def load_fonts():
    """Register the bundled OFL fonts with matplotlib (once)."""
    global _fonts_loaded
    if _fonts_loaded:
        return
    import logging
    import matplotlib.font_manager as fm
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    d = resource("fonts")
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith((".ttf", ".otf")):
                try:
                    fm.fontManager.addfont(os.path.join(d, f))
                except Exception:
                    pass
    _fonts_loaded = True


def load_themes():
    """Built-in themes, overridden/extended by %APPDATA%/MapComposer/themes.yaml."""
    out = {}
    for p in (resource("themes.yaml"), os.path.join(user_dir(), "themes.yaml")):
        if os.path.exists(p):
            mt = os.path.getmtime(p)
            if _theme_cache.get(p, (None,))[0] != mt:
                with open(p, encoding="utf-8") as f:
                    _theme_cache[p] = (mt, yaml.safe_load(f) or {})
            for k, v in _theme_cache[p][1].items():
                base = dict(out.get(k) or out.get("classic") or {})
                base.update(v or {})
                base["id"] = k
                out[k] = base
    return out
