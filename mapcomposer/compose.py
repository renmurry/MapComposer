"""Map composer: draws one map sheet from a template, rasters and vectors."""
import math
import re
import textwrap
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize, to_rgb
from matplotlib.patches import Rectangle, Polygon as MPoly
from matplotlib.lines import Line2D
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio.warp import transform_bounds
from shapely.geometry import Point, box
from shapely.ops import unary_union

from .geom import SQFT, straighten, explode_polys, ring_edges
from .styles import load_styles, class_colors, is_lulc_values, pretty, match_rule
from .place import place_line, fill_text

PAPERS = {"A3": (420.0, 297.0), "A4": (297.0, 210.0), "A2": (594.0, 420.0), "A1": (841.0, 594.0)}
NICE_SCALES = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000,
               7500, 10000, 12500, 15000, 20000, 25000, 30000, 40000, 50000, 75000, 100000, 150000, 200000,
               250000, 300000, 500000, 750000, 1000000, 2000000]
DARK, NAVY = "#1a1a1a", "#1f3b57"
SOFT = ["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc", "#e5d8bd", "#fddaec", "#b3e2cd",
        "#fdcdac", "#cbd5e8", "#f4cae4", "#e6f5c9", "#fff2ae", "#f1e2cc"]
POSITIONING = {
    "gnss": "Positioning: drone GNSS only (no GCP / RTK). Absolute coordinates may be offset by a few metres",
    "rtk": "Positioning: RTK/PPK-corrected drone positions",
    "gcp": "Positioning: ground control points",
    "none": "",
}


@dataclass
class Job:
    template: dict
    rasters: list
    vectors: list
    out_base: str
    texts: dict = field(default_factory=dict)       # name,title,subtitle,date,source,positioning,rmse,prepared_by,disclaimer
    paper: str = ""
    orientation: str = ""
    scale: int = 0
    side_lengths: bool = None
    straighten: bool = False
    straighten_tol: float = 0.15
    contours: bool = None
    contour_interval: float = 0
    png_dpi: int = 300
    write_pdf: bool = True
    write_png: bool = True
    place_index: object = None
    map_date: str = field(default_factory=lambda: date.today().strftime("%d %b %Y"))


def _nat(x):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(x))]


def _halo(w, c="white"):
    return [pe.withStroke(linewidth=w, foreground=c)]


def _nice_le(v, choices):
    best = choices[0]
    for c in choices:
        if c <= v:
            best = c
    return best


def _fmt_area(m2, unit):
    if unit == "m2":
        return f"{m2:,.2f} m²"
    if unit == "ha":
        return f"{m2 / 1e4:,.2f} ha"
    return f"{m2 / 1e6:,.3f} km²"


def _area_unit(max_m2):
    return "m2" if max_m2 < 1e4 else ("ha" if max_m2 < 5e6 else "km2")


def _fmt_len(m, unit):
    return f"{m:,.1f} m" if unit == "m" else f"{m / 1000:,.2f} km"


def _dms(v, pos, neg):
    h = pos if v >= 0 else neg
    v = abs(v); d = int(v); m = int((v - d) * 60); s = round(((v - d) * 60 - m) * 60)
    if s == 60:
        s = 0; m += 1
    if m == 60:
        m = 0; d += 1
    return f"{d}°{m:02d}'{s:02d}\"{h}" if s else f"{d}°{m:02d}'{h}"


def _stretch(arr, mask):
    out = np.zeros(arr.shape, dtype=np.uint8)
    for i in range(arr.shape[0]):
        b = arr[i].astype("float64")
        v = b[mask] if mask.any() else b.ravel()
        if arr.dtype == np.uint8 and arr.shape[0] == 3 and v.size and np.percentile(v, 99) > 150:
            out[i] = arr[i]; continue
        lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
        out[i] = np.clip((b - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    return out


def _nan_blur(a, k):
    if k <= 1:
        return a
    v = np.where(np.isnan(a), 0.0, a); m = (~np.isnan(a)).astype(float); p = k // 2

    def bx(x):
        x = np.pad(x, p, mode="edge"); c = np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    s, n = bx(v), bx(m)
    out = np.where(n > 0, s / np.maximum(n, 1e-9), np.nan)[:a.shape[0], :a.shape[1]]
    out[np.isnan(a)] = np.nan
    return out


class Page:
    """Layout units are A3-equivalent millimetres; S scales them to the real paper."""

    def __init__(self, paper, orientation):
        w, h = PAPERS[paper]
        if orientation == "portrait":
            w, h = h, w
        self.paper, self.orient = paper, orientation
        self.S = max(w, h) / 420.0
        self.W, self.H = w / self.S, h / self.S
        self.fig = plt.figure(figsize=(w / 25.4, h / 25.4))

    def x(self, v): return v / self.W
    def y(self, v): return v / self.H
    def fs(self, pt): return pt * self.S

    def rect(self, x0, y0, w, h, **kw):
        kw.setdefault("fill", False)
        self.fig.add_artist(Rectangle((self.x(x0), self.y(y0)), self.x(w), self.y(h), transform=self.fig.transFigure, **kw))

    def text(self, x, y, s, size, **kw):
        return self.fig.text(self.x(x), self.y(y), s, fontsize=self.fs(size), **kw)

    def axes(self, x0, y0, w, h):
        return self.fig.add_axes([self.x(x0), self.y(y0), self.x(w), self.y(h)])


# =====================================================================================
def compose(job: Job, preview=False, log=print):
    T = job.template
    st = load_styles()
    rep = dict(plots=[], warnings=[], outputs=[], skipped_edges=[], scale=None, crs=None, places={})
    warn = rep["warnings"]
    rasters = [r for r in job.rasters if r["role"] != "Ignore"]
    vectors = [v for v in job.vectors if v["role"] != "Ignore" and len(v["layer"])]
    if not rasters and not vectors:
        raise ValueError("Nothing to draw: every layer is set to Ignore.")

    # ---------------- map CRS ----------------
    crs, auto = None, False
    for role in ("Basemap", "Classified", "Elevation", "Continuous"):
        for r in rasters:
            c = r["layer"].crs
            if r["role"] == role and c is not None and c.is_projected and c.axis_info[0].unit_name in ("metre", "meter"):
                crs = c; break
        if crs:
            break
    if crs is None:
        for v in vectors:
            c = v["layer"].crs
            if c is not None and CRS(c).is_projected and CRS(c).axis_info[0].unit_name in ("metre", "meter"):
                crs = CRS(c); break
    if crs is None:
        pts = []
        for v in vectors:
            if v["layer"].crs is not None:
                pts.append(v["layer"].gdf.to_crs(4326).union_all().centroid)
        for r in rasters:
            if r["layer"].crs is not None:
                b = transform_bounds(r["layer"].crs.to_wkt(), "EPSG:4326", *r["layer"].bounds)
                pts.append(Point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        if not pts:
            raise ValueError("No layer has a coordinate system - cannot place the map.")
        lon, lat = pts[0].x, pts[0].y
        info = query_utm_crs_info(datum_name="WGS 84", area_of_interest=AreaOfInterest(lon, lat, lon, lat))
        crs = CRS.from_epsg(info[0].code); auto = True
        log(f"Map projection chosen automatically: {crs.name}")
    rep["crs"] = crs.name

    gdfs = {}
    for i, v in enumerate(vectors):
        if v["layer"].crs is None:
            warn.append(f"{v['layer'].name}: no coordinate system, assumed {crs.name}")
        gdfs[i] = v["layer"].to_map(crs)

    def raster_bounds(r):
        rl = r["layer"]
        if rl.crs is None:
            return rl.bounds
        return transform_bounds(rl.crs.to_wkt(), crs.to_wkt(), *rl.bounds)

    # ---------------- plots (Boundary) ----------------
    plots = []
    for i, v in enumerate(vectors):
        if v["role"] != "Boundary":
            continue
        g = gdfs[i]
        for k, geom in enumerate(g.geometry):
            parts = explode_polys(geom)
            for j, p in enumerate(parts):
                if job.straighten:
                    p2, removed = straighten(p, job.straighten_tol)
                    if removed:
                        log(f"Straightened: removed {len(removed)} corner(s), offsets {removed} m; "
                            f"area {p.area:,.2f} -> {p2.area:,.2f} m²")
                    p = p2
                lbl = ""
                if v["label_field"]:
                    lbl = str(g.iloc[k][v["label_field"]]).strip()
                    if len(parts) > 1:
                        lbl = f"{lbl}-{j + 1}"
                plots.append(dict(poly=p, label=lbl))
    if job.straighten and len(plots) > 1:
        warn.append("Straightening was applied to each plot separately - check shared edges still line up")
    if len(plots) == 1:
        plots[0]["label"] = plots[0]["label"] or job.texts.get("name", "")
    else:
        for k, p in enumerate(plots):
            p["label"] = p["label"] or f"Plot {chr(65 + k) if k < 26 else k + 1}"
    aoi = unary_union([p["poly"] for p in plots]) if plots else None

    # ---------------- extent ----------------
    ex = T["extent"]
    geoms_b = []
    for role in ex.get("roles", []):
        got = [gdfs[i].total_bounds for i, v in enumerate(vectors) if v["role"] == role and role != "Boundary"]
        if role == "Boundary" and plots:
            got = [aoi.bounds]
        geoms_b += got
        if geoms_b and ex.get("mode", "first") == "first":
            break
    if not geoms_b:
        geoms_b = [raster_bounds(r) for r in rasters if r["layer"].kind in ex.get("rasters", [])]
    if not geoms_b:
        geoms_b = [gdfs[i].total_bounds for i in range(len(vectors))]
    if not geoms_b:
        geoms_b = [raster_bounds(r) for r in rasters]
    minx = min(b[0] for b in geoms_b); miny = min(b[1] for b in geoms_b)
    maxx = max(b[2] for b in geoms_b); maxy = max(b[3] for b in geoms_b)
    bw, bh = max(maxx - minx, 1.0), max(maxy - miny, 1.0)

    # ---------------- page ----------------
    paper = job.paper or T["page"]["paper"]
    orient = job.orientation or T["page"]["orientation"]
    pg = Page(paper, orient)
    S = pg.S
    pg.rect(5, 5, pg.W - 10, pg.H - 10, lw=1.6 * S, ec="#222")
    pg.rect(6.5, 6.5, pg.W - 13, pg.H - 13, lw=0.5 * S, ec="#222")
    if orient == "landscape":
        mx0, my0, mw, mh = 20, 17, pg.W - 141, pg.H - 31
        columns = [(pg.W - 114, 8, 104, pg.H - 22)]
        panel = [T["panel"]]
    else:
        mx0, my0, mw, mh = 20, 140, pg.W - 35, pg.H - 154
        cw = (mw - 6) / 3
        columns = [(mx0 + i * (cw + 3), 8, cw, 124) for i in range(3)]
        panel = T["panel_portrait"]
    pmw, pmh = mw * S, mh * S
    margin = ex.get("margin_mm", 18) * S
    need = max(bw / ((pmw - 2 * margin) / 1000), bh / ((pmh - 2 * margin) / 1000), 1)
    denom = job.scale or next((s for s in NICE_SCALES if s >= need), int(math.ceil(need / 1e5) * 1e5))
    if job.scale and job.scale < need:
        warn.append(f"Scale 1:{job.scale:,} is too large to fit everything; about 1:{int(need):,} or smaller would fit")
    rep["scale"] = denom
    mpp = denom / 1000.0
    gw, gh = pmw * mpp, pmh * mpp
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    ext = (cx - gw / 2, cx + gw / 2, cy - gh / 2, cy + gh / 2)
    frame = box(ext[0], ext[2], ext[1], ext[3])

    ax = pg.axes(mx0, my0, mw, mh)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    max_px = int(pmw / 25.4 * (110 if preview else max(job.png_dpi, 250)))
    legend, notes_extra = [], []

    # ---------------- rasters ----------------
    imagery = None
    basemap = next((r for r in rasters if r["role"] == "Basemap"), None) if T.get("basemap", True) else None
    plain = True
    if basemap:
        rl = basemap["layer"]
        arr, mask, iext = rl.read(crs, ext, max_px, bands=basemap["bands"])
        if arr is not None:
            rgb = _stretch(arr, mask)
            img = np.dstack([rgb[0], rgb[1], rgb[2], np.where(mask, 255, 0).astype(np.uint8)])
            ax.imshow(img, extent=iext, origin="upper", interpolation="lanczos", zorder=0)
            plain = False
            imagery = rl
    ax.set_facecolor("#ebebeb" if not plain else "#f7f7f4")

    elev = next((r for r in rasters if r["role"] == "Elevation"), None)
    if elev and (plain or T["id"] == "terrain"):
        _draw_elevation(ax, elev, crs, ext, max_px, st, legend, T, S)
        plain = False
    for r in rasters:
        if r["role"] == "Continuous" and plain:
            _draw_continuous(ax, r, crs, ext, max_px, st, legend)
            plain = False
    clip_geom = aoi if (T.get("clip_to_boundary") and aoi is not None) else None
    for r in rasters:
        if r["role"] == "Classified":
            _draw_classified(ax, r, crs, ext, max_px, st, legend, alpha=1.0 if plain else 0.8, clip=clip_geom)
            plain = False

    do_cont = T["contours"]["default"] if job.contours is None else job.contours
    civ = job.contour_interval or T["contours"]["interval"]
    if do_cont and elev:
        try:
            civ = _draw_contours(ax, elev, crs, ext, civ, legend, warn, S, preview, on_image=not (imagery is None and T["id"] == "terrain"), st=st)
            notes_extra.append(("contours", f"Contours: {civ:g} m interval from {elev['layer'].name} "
                                            "(DSM includes buildings and trees)" if elev["layer"].name.lower().find("dsm") >= 0
                                 else f"Contours: {civ:g} m interval from {elev['layer'].name}"))
        except Exception as e:
            warn.append(f"Contours skipped: {e}")

    # ---------------- vectors ----------------
    order = {"Thematic": 1, "Admin": 2, "Contour": 3, "Drainage": 4, "Building": 5, "Other": 6, "Road": 7, "Points": 9}
    for i in sorted([i for i, v in enumerate(vectors) if v["role"] != "Boundary"], key=lambda i: order.get(vectors[i]["role"], 8)):
        g_i = gdfs[i]
        if clip_geom is not None and vectors[i]["role"] == "Thematic":
            g_i = g_i.clip(clip_geom)
        _draw_vector(ax, vectors[i], g_i, frame, legend, st, S, plain, T, mpp, pg)

    # ---------------- boundaries ----------------
    bopt = T["boundary"]
    sides = bopt.get("side_lengths") if job.side_lengths is None else job.side_lengths
    bst = st["roles"]["Boundary"]
    lw = bst["width"] * S
    labelled = set()
    for p in plots:
        poly = p["poly"]
        for ring_i, ring in enumerate([poly.exterior] + list(poly.interiors)):
            xs, ys = ring.xy
            if ring_i == 0 and bopt.get("fill", False):
                ax.add_patch(MPoly(np.c_[xs, ys], closed=True, fc=(*to_rgb(bst["fill"]), bst["fill_alpha"]), ec="none", zorder=20))
            ax.plot(xs, ys, color=bst["edge"], lw=lw, zorder=21, ls="-" if ring_i == 0 else "--", path_effects=_halo(lw + 2.2 * S, bst["halo"]))
        xs, ys = poly.exterior.xy
        if bopt.get("corners"):
            ax.scatter(xs[:-1], ys[:-1], s=22 * S * S, c="white", edgecolors=bst["edge"], linewidths=1.3 * S, zorder=22)
        if sides:
            for a, bb, Ln in ring_edges(poly.exterior.coords):
                key = tuple(sorted([(round(a[0], 2), round(a[1], 2)), (round(bb[0], 2), round(bb[1], 2))]))
                if key in labelled:
                    continue
                labelled.add(key)
                if Ln / mpp < 9:
                    rep["skipped_edges"].append(round(Ln, 2)); continue
                mxp, myp = (a[0] + bb[0]) / 2, (a[1] + bb[1]) / 2
                dx, dy = (bb[0] - a[0]) / Ln, (bb[1] - a[1]) / Ln
                nx, ny = -dy, dx
                off = 6.5 * S * mpp
                if poly.contains(Point(mxp + nx * off, myp + ny * off)):
                    nx, ny = -nx, -ny
                ang = math.degrees(math.atan2(dy, dx))
                ang = ang - 180 if ang > 90 else (ang + 180 if ang < -90 else ang)
                ax.text(mxp + nx * off, myp + ny * off, f"{Ln:.2f} m", rotation=ang, rotation_mode="anchor",
                        ha="center", va="center", fontsize=pg.fs(9.5), fontweight="bold", color="#111", zorder=30,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=bst["edge"], lw=0.8 * S, alpha=0.92))
        a = poly.area
        if bopt.get("area_label"):
            lp = poly.centroid if poly.contains(poly.centroid) else poly.representative_point()
            head = p["label"].upper() if p["label"] else ""
            unit = _area_unit(a)
            atxt = f"{a:,.2f} m²\n{a * SQFT:,.2f} sq ft" if unit == "m2" else _fmt_area(a, unit)
            ax.text(lp.x, lp.y, (f"{head}\n" if head else "") + atxt, ha="center", va="center",
                    fontsize=pg.fs(12 if len(plots) == 1 else 9.5), fontweight="bold", color="white", zorder=31,
                    linespacing=1.3, path_effects=_halo(3.2 * S, "#000"))
        rep["plots"].append(dict(label=p["label"] or "Plot", area=a, sqft=a * SQFT, perimeter=poly.length))
    if rep["skipped_edges"]:
        warn.append(f"Side labels left off {len(rep['skipped_edges'])} very short edge(s): {rep['skipped_edges']} m")

    # ---------------- grid ----------------
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_xlabel(""); ax.set_ylabel("")
    if T["grid"] == "latlon":
        _graticule(ax, crs, ext, pg, plain)
    elif T["grid"] == "utm":
        _utm_grid(ax, ext, gw, mpp, pg, plain)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.2 * S)

    # ---------------- place context ----------------
    center_pt = (aoi.centroid if aoi is not None else Point(cx, cy))
    places = {}
    if job.place_index:
        try:
            places = job.place_index.lookup(center_pt, crs)
        except Exception as e:
            warn.append(f"Place lookup failed: {e}")
    rep["places"] = places
    vals = dict(places)
    vals.update(name=job.texts.get("name", ""), place_line=place_line(places))
    title = fill_text(job.texts.get("title") or T["text"]["title"], vals) or job.texts.get("name", "Map")
    subtitle = fill_text(job.texts.get("subtitle") if job.texts.get("subtitle") is not None else T["text"]["subtitle"], vals)
    rep["title"] = title

    # ---------------- stats ----------------
    stats_title, rows = _stats(T["stats"], plots, rep, vectors, gdfs, rasters, crs, aoi, frame, st, warn)

    # ---------------- notes ----------------
    notes = []
    for key in T["notes"]:
        if key == "crs":
            epsg = crs.to_epsg()
            notes.append(f"Coordinate system: {crs.name}" + (f" (EPSG:{epsg})" if epsg else "") +
                         (" - chosen automatically from the data location" if auto else ""))
        elif key == "imagery" and imagery is not None:
            res = imagery.res_m
            dt = job.texts.get("date", "")
            if res < 0.5:
                s = f"Imagery: UAV orthomosaic, ~{res * 100:.1f} cm GSD" + (f", flown {dt}" if dt else "")
            else:
                s = f"Imagery: {imagery.name}, {res:g} m resolution" + (f", acquired {dt}" if dt else "")
            if imagery.kind == "multiband":
                s += f", bands {'-'.join(map(str, basemap['bands']))} as RGB"
            notes.append(s)
        elif key == "elevation_source" and elev:
            dt = job.texts.get("date", "")
            notes.append(f"Elevation: {elev['layer'].name}, {elev['layer'].res_m:.2g} m cells" + (f", {dt}" if dt else ""))
        elif key == "source" and job.texts.get("source"):
            notes.append("Data source: " + job.texts["source"])
        elif key == "positioning":
            pos = POSITIONING.get(job.texts.get("positioning", "gnss"), "")
            if pos:
                if job.texts.get("positioning") == "gcp" and job.texts.get("rmse"):
                    pos += f", RMSE {job.texts['rmse']}"
                notes.append(pos)
        elif key == "measurements" and (plots or T["stats"] in ("admin_areas", "class_areas", "network_lengths")):
            notes.append("Areas and lengths are planimetric, computed in the map coordinate system")
        elif key == "contours":
            notes += [t for k, t in notes_extra if k == "contours"]
        elif key == "north_date":
            notes.append(f"North arrow shows grid north  |  Map date: {job.map_date}")
        elif key == "disclaimer" and job.texts.get("disclaimer"):
            notes.append(job.texts["disclaimer"])
    if "contours" not in T["notes"]:
        notes += [t for k, t in notes_extra if k == "contours"]

    # ---------------- panel ----------------
    ctx = dict(pg=pg, S=S, title=title, subtitle=subtitle, denom=denom, paper=paper, legend=legend, plots=plots,
               T=T, rows=rows, stats_title=stats_title, notes=notes, prepared_by=job.texts.get("prepared_by", ""),
               mpp=mpp, ext=ext, gw=gw, gh=gh, crs=crs, rasters=rasters, basemap=basemap, imagery=imagery,
               place_index=job.place_index, center_pt=center_pt, aoi=aoi, preview=preview, places=places,
               vectors=vectors, gdfs=gdfs)
    _draw_panel(ctx, columns, panel)

    # ---------------- save ----------------
    meta = {"Title": title, "Author": job.texts.get("prepared_by") or None, "Creator": "MapComposer", "Producer": "MapComposer"}
    if preview:
        out = job.out_base + "_preview.png"
        pg.fig.savefig(out, dpi=90, metadata={"Software": "MapComposer"}); rep["outputs"].append(out)
    else:
        if job.write_pdf:
            pg.fig.savefig(job.out_base + ".pdf", dpi=300, metadata=meta); rep["outputs"].append(job.out_base + ".pdf")
        if job.write_png:
            pg.fig.savefig(job.out_base + ".png", dpi=job.png_dpi, metadata={"Software": "MapComposer"})
            rep["outputs"].append(job.out_base + ".png")
    plt.close(pg.fig)
    return rep


# =====================================================================================
# raster drawing
def _draw_elevation(ax, r, crs, ext, max_px, st, legend, T, S):
    rl = r["layer"]
    arr, mask, iext = rl.read(crs, ext, min(max_px, 2200))
    if arr is None:
        return
    z = arr[0].astype("float64")
    z[~mask] = np.nan
    z[z < -1000] = np.nan
    if np.all(np.isnan(z)):
        return
    lo, hi = np.nanpercentile(z, [1, 99])
    cmap = LinearSegmentedColormap.from_list("elev", st["elevation_cmap"])
    norm = Normalize(lo, hi)
    filled = np.where(np.isnan(z), np.nanmean(z), z)
    if T.get("elevation", {}).get("hillshade", True):
        cell = (iext[1] - iext[0]) / z.shape[1]
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(filled, cmap=cmap, norm=norm, blend_mode="soft", vert_exag=1.0, dx=cell, dy=cell)
    else:
        rgb = cmap(norm(filled))
    rgb[..., 3] = np.where(np.isnan(z), 0, T.get("elevation", {}).get("opacity", 1.0))
    ax.imshow(rgb, extent=iext, origin="upper", zorder=1, interpolation="bilinear")
    legend.append(dict(kind="colorbar", cmap=cmap, vmin=lo, vmax=hi, label="Elevation (m)"))


def _draw_continuous(ax, r, crs, ext, max_px, st, legend):
    arr, mask, iext = r["layer"].read(crs, ext, max_px)
    if arr is None:
        return
    z = np.ma.masked_array(arr[0].astype("float64"), ~mask)
    lo, hi = np.percentile(z.compressed(), [2, 98]) if z.count() else (0, 1)
    cmap = plt.get_cmap(st.get("continuous_cmap", "viridis"))
    ax.imshow(z, extent=iext, origin="upper", cmap=cmap, vmin=lo, vmax=hi, zorder=1)
    legend.append(dict(kind="colorbar", cmap=cmap, vmin=lo, vmax=hi, label=pretty(r["layer"].name)))


def _class_table(rl, values, st):
    classes = dict(rl.classes or {})
    pal = st["fallback_palette"]; k = 0
    out = {}
    for v in sorted(values):
        c = classes.get(int(v), {})
        name = c.get("name") or f"Class {int(v)}"
        col = c.get("color")
        if not col:
            rule = match_rule(st["lulc"], name)
            col = rule["color"] if rule else pal[k % len(pal)]; k += 1
        out[int(v)] = dict(name=name, color=col)
    return out


def _draw_classified(ax, r, crs, ext, max_px, st, legend, alpha=1.0, clip=None):
    rl = r["layer"]
    arr, mask, iext = rl.read(crs, ext, max_px)
    if arr is None:
        return
    a = arr[0]
    if clip is not None:
        from rasterio.features import geometry_mask
        from rasterio.transform import from_bounds as tfb
        tr = tfb(iext[0], iext[2], iext[1], iext[3], a.shape[1], a.shape[0])
        mask = mask & geometry_mask([clip], out_shape=a.shape, transform=tr, invert=True)
    vals = np.unique(a[mask])
    if rl.nodata is not None:
        vals = vals[vals != rl.nodata]
    if len(vals) > 60:
        raise ValueError(f"{rl.name}: {len(vals)} classes - set its role to Continuous instead")
    table = _class_table(rl, vals, st)
    rgba = np.zeros(a.shape + (4,), dtype=float)
    for v, c in table.items():
        sel = (a == v) & mask
        rgba[sel, :3] = to_rgb(c["color"]); rgba[sel, 3] = alpha
    ax.imshow(rgba, extent=iext, origin="upper", interpolation="nearest", zorder=2)
    for v, c in table.items():
        legend.append(dict(kind="patch", label=c["name"], fc=c["color"], ec="#555", alpha=1.0))
    r["_table"] = table


def _draw_contours(ax, r, crs, ext, iv, legend, warn, S, preview, on_image, st):
    rl = r["layer"]
    target = max(rl.res_m, (ext[1] - ext[0]) / (250 if preview else 700))
    arr, mask, iext = rl.read(crs, ext, int((ext[1] - ext[0]) / target))
    if arr is None:
        raise ValueError("DSM does not cover the map area")
    a = arr[0].astype("float64"); a[~mask] = np.nan; a[a < -1000] = np.nan
    if np.all(np.isnan(a)):
        raise ValueError("DSM has no data in the map area")
    cell = (iext[1] - iext[0]) / a.shape[1]
    a = _nan_blur(a, max(1, int(round(3.0 / cell))) | 1)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if (hi - lo) / iv > 150:
        iv = _nice_le((hi - lo) / 75, [0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200])
        warn.append(f"Contour interval raised to {iv:g} m (too many lines)")
    levels = np.arange(math.floor(lo / iv) * iv, hi + iv, iv)
    xs = np.linspace(iext[0], iext[1], a.shape[1]); ys = np.linspace(iext[3], iext[2], a.shape[0])
    is_major = (np.round(levels / iv).astype(int) % 5) == 0
    col = st["roles"]["Contour"]["color"] if on_image else st["roles"]["Contour"]["color_plain"]
    if (~is_major).any():
        ax.contour(xs, ys, a, levels=levels[~is_major], colors=col, linewidths=0.5 * S, alpha=0.85, zorder=5)
    if is_major.any():
        cs = ax.contour(xs, ys, a, levels=levels[is_major], colors=col, linewidths=1.0 * S, alpha=0.95, zorder=5)
        ax.clabel(cs, fmt="%g", fontsize=5.5 * S, colors=col)
    legend.append(dict(kind="line", label=f"Contour ({iv:g} m)", color=col, lw=1.0 * S, halo=DARK if on_image else None))
    return iv


# =====================================================================================
# vector drawing
def _draw_vector(ax, v, g, frame, legend, st, S, plain, T, mpp, pg):
    role, lyr, sf = v["role"], v["layer"], v["style_field"]
    rs = st["roles"].get(role, st["roles"]["Other"])
    g = g[g.intersects(frame)]
    if not len(g):
        return
    # class values & colours
    if sf and sf in g.columns:
        cls = g[sf].astype(str)
        uniq = sorted(cls.unique(), key=_nat)
        if role == "Road":
            cmap = class_colors("road", uniq)
        elif role == "Thematic" or (role == "Other" and is_lulc_values(uniq)):
            cmap = class_colors("lulc", uniq)
        elif role == "Drainage":
            cmap = None
        else:
            cmap = class_colors("generic", uniq)
    else:
        cls, uniq, cmap = None, [""], None

    if role == "Drainage":
        dst = st["drainage_order"]
        if sf and sf in g.columns:
            try:
                orders = g[sf].astype(float)
            except ValueError:
                orders = None
        else:
            orders = None
        if orders is not None:
            for o in sorted(orders.unique()):
                sub = g[orders == o]
                w = (dst["base_width"] + dst["per_order"] * o) * S
                sub.plot(ax=ax, color=dst["color"], linewidth=w, zorder=10,
                         path_effects=[pe.withStroke(linewidth=w + 1.4 * S, foreground=rs["halo"])])
                legend.append(dict(kind="line", label=f"Stream order {o:g}", color=dst["color"], lw=w, halo=rs["halo"]))
            return
        g.plot(ax=ax, color=rs["color"], linewidth=rs["width"] * S, zorder=10,
               path_effects=[pe.withStroke(linewidth=rs["width"] * S + 1.6 * S, foreground=rs["halo"])])
        legend.append(dict(kind="line", label="Drainage", color=rs["color"], lw=rs["width"] * S, halo=rs["halo"]))
        return

    for u in uniq:
        sub = g if cls is None else g[cls == u]
        sty = cmap[u] if cmap else {}
        kinds = set(t.replace("Multi", "") for t in sub.geom_type)
        if role == "Admin":
            edge = rs["edge_plain"] if plain else rs["edge"]
            lwid = rs["width"] * S
            if plain:
                cols = [SOFT[k % len(SOFT)] for k in range(len(sub))]
                sub.plot(ax=ax, color=cols, alpha=rs["fill_alpha_plain"], edgecolor="none", zorder=3)
            sub.boundary.plot(ax=ax, color=edge, linewidth=lwid, zorder=11,
                              path_effects=None if plain else [pe.withStroke(linewidth=lwid + 1.8 * S, foreground=rs["halo"])])
            lab = pretty(u) if u else pretty(lyr.name)
            legend.append(dict(kind="patch", label=lab, fc="#f0e6d2" if plain else "none", ec=edge, lw=lwid,
                               halo=None if plain else rs["halo"]))
            if T["labels"].get("Admin", True) and v["label_field"]:
                _poly_labels(ax, sub, v["label_field"], mpp, pg, plain)
            continue
        col = sty.get("color") or rs.get("color") or rs.get("edge") or "#b6ff3b"
        width = (sty.get("width") or rs.get("width", 1.6)) * S
        halo = rs.get("halo", DARK)
        name = pretty(u) if u else {"Road": "Road", "Building": "Building", "Points": pretty(lyr.name),
                                    "Contour": "Contour", "Other": pretty(lyr.name), "Thematic": pretty(lyr.name)}[role]
        if role == "Other" and u:
            name = f"{pretty(lyr.name)}: {pretty(u)}"
        if "Polygon" in kinds:
            if role == "Thematic":
                fa = rs["fill_alpha"] if plain else rs["fill_alpha_on_image"]
                sub.plot(ax=ax, color=col, alpha=fa, edgecolor="none", zorder=3)
                sub.boundary.plot(ax=ax, color=rs["edge"], linewidth=rs["width"] * S, alpha=0.6, zorder=4)
                legend.append(dict(kind="patch", label=name, fc=col, ec="#555", alpha=1.0))
            else:
                fa = rs.get("fill_alpha", 0.12)
                sub.plot(ax=ax, color=col, alpha=fa, edgecolor="none", zorder=9)
                sub.boundary.plot(ax=ax, color=col, linewidth=width, zorder=10,
                                  path_effects=[pe.withStroke(linewidth=width + 1.8 * S, foreground=halo)])
                legend.append(dict(kind="patch", label=name, fc=col, ec=col, alpha=0.35, halo=halo, lw=1.5 * S))
        if "LineString" in kinds:
            sub.plot(ax=ax, color=col, linewidth=width, zorder=10, linestyle="--" if sty.get("dash") else "-",
                     path_effects=[pe.withStroke(linewidth=width + 2.2 * S, foreground=halo)])
            legend.append(dict(kind="line", label=name, color=col, lw=width, halo=halo, dash=sty.get("dash")))
        if "Point" in kinds:
            pc = sty.get("color") or rs.get("color", "#ff2020")
            ax.scatter(sub.geometry.x, sub.geometry.y, s=(rs.get("size", 7) * S) ** 2, c=pc,
                       edgecolors=rs.get("edge", "white"), linewidths=1 * S, zorder=25)
            legend.append(dict(kind="marker", label=name, color=pc, ec=rs.get("edge", "white")))
            if T["labels"].get("Points", True) and v["label_field"]:
                for _, row in sub.iterrows():
                    ax.annotate(str(row[v["label_field"]]), (row.geometry.x, row.geometry.y), xytext=(4 * S, 4 * S),
                                textcoords="offset points", fontsize=pg.fs(7), color="white" if not plain else "#111",
                                zorder=26, path_effects=_halo(2 * S, "#000" if not plain else "white"))


def _poly_labels(ax, g, field_, mpp, pg, plain):
    for _, row in g.iterrows():
        geom = row.geometry
        area_mm2 = geom.area / (mpp * mpp)
        if area_mm2 < 40:
            continue
        fs = min(10.0, max(5.0, math.sqrt(area_mm2) / 5.5))
        p = geom.representative_point()
        ax.text(p.x, p.y, textwrap.fill(str(row[field_]), 16), ha="center", va="center", fontsize=pg.fs(fs),
                color="#222" if plain else "white", fontweight="bold", zorder=24,
                path_effects=_halo(2.4 * pg.S, "white" if plain else "#000"))


# =====================================================================================
# grids
def _utm_grid(ax, ext, gw, mpp, pg, plain):
    S = pg.S
    steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000, 100000]
    step = next((s for s in steps if gw / s <= 9), steps[-1])
    gx = np.arange(math.ceil(ext[0] / step) * step, ext[1], step)
    gy = np.arange(math.ceil(ext[2] / step) * step, ext[3], step)
    cl = 0.8 * S * mpp
    col = "#666" if plain else "white"
    for x in gx:
        for y in gy:
            ax.plot([x - cl, x + cl], [y, y], color=col, lw=0.7 * S, alpha=0.8, zorder=15)
            ax.plot([x, x], [y - cl, y + cl], color=col, lw=0.7 * S, alpha=0.8, zorder=15)
    ax.set_xticks(gx); ax.set_yticks(gy)
    ax.set_xticklabels([f"{v:.0f} E" for v in gx], fontsize=pg.fs(6.5))
    ax.set_yticklabels([f"{v:.0f} N" for v in gy], fontsize=pg.fs(6.5), rotation=90, va="center")
    ax.tick_params(direction="in", length=4 * S, width=0.6 * S, top=True, right=True, labeltop=True,
                   labelright=False, pad=2 * S)


def _graticule(ax, crs, ext, pg, plain):
    S = pg.S
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_m = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs = np.linspace(ext[0], ext[1], 30); ys = np.linspace(ext[2], ext[3], 30)
    bx = np.r_[xs, np.full(30, ext[1]), xs[::-1], np.full(30, ext[0])]
    by = np.r_[np.full(30, ext[2]), ys, np.full(30, ext[3]), ys[::-1]]
    lon, lat = to_ll.transform(bx, by)
    lo0, lo1, la0, la1 = min(lon), max(lon), min(lat), max(lat)
    secs = [1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200, 18000, 36000]
    step = next((s for s in secs if (lo1 - lo0) * 3600 / s <= 5), secs[-1]) / 3600.0
    col = "#888" if plain else "white"
    xt, xl, yt, yl = [], [], [], []
    for L in np.arange(math.ceil(lo0 / step) * step, lo1, step):
        la = np.linspace(la0, la1, 60); x, y = to_m.transform(np.full(60, L), la)
        ax.plot(x, y, color=col, lw=0.5 * S, alpha=0.6, zorder=15)
        xb = np.interp(ext[2], y, x)
        if ext[0] + 0.03 * (ext[1] - ext[0]) < xb < ext[1] - 0.03 * (ext[1] - ext[0]):
            xt.append(xb); xl.append(_dms(L, "E", "W"))
    for L in np.arange(math.ceil(la0 / step) * step, la1, step):
        lo = np.linspace(lo0, lo1, 60); x, y = to_m.transform(lo, np.full(60, L))
        ax.plot(x, y, color=col, lw=0.5 * S, alpha=0.6, zorder=15)
        yb = np.interp(ext[0], x, y)
        if ext[2] + 0.03 * (ext[3] - ext[2]) < yb < ext[3] - 0.03 * (ext[3] - ext[2]):
            yt.append(yb); yl.append(_dms(L, "N", "S"))
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=pg.fs(6.5))
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=pg.fs(6.5), rotation=90, va="center")
    ax.tick_params(direction="in", length=4 * S, width=0.6 * S, top=True, right=True, labeltop=True,
                   labelright=False, pad=2 * S)


# =====================================================================================
# statistics
def _stats(kind, plots, rep, vectors, gdfs, rasters, crs, aoi, frame, st, warn):
    rows = []
    if kind == "plot_details" and plots:
        if len(plots) == 1:
            pr = rep["plots"][0]; c = plots[0]["poly"].centroid
            lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(c.x, c.y)
            u = _area_unit(pr["area"])
            rows = [("Area", _fmt_area(pr["area"], u))]
            if u == "m2":
                rows.append(("", f"{pr['sqft']:,.2f} sq ft"))
            rows += [("Perimeter", f"{pr['perimeter']:,.2f} m"), ("Centroid (E, N)", f"{c.x:,.2f} E"),
                     ("", f"{c.y:,.2f} N"), ("Centroid (Lat/Long)", f"{lat:.6f}° N, {lon:.6f}° E")]
            return "PLOT DETAILS", rows
        u = _area_unit(max(p["area"] for p in rep["plots"]))
        for p in rep["plots"]:
            rows.append((p["label"], _fmt_area(p["area"], u) + (f"  ({p['sqft']:,.1f} sq ft)" if u == "m2" else "")))
        tot = sum(p["area"] for p in rep["plots"])
        rows.append(("Total", _fmt_area(tot, u) + (f"  ({tot * SQFT:,.1f} sq ft)" if u == "m2" else "")))
        return "PLOT AREAS", rows
    if kind == "admin_areas":
        items = []
        for i, v in enumerate(vectors):
            if v["role"] == "Admin":
                g = gdfs[i]
                for _, row in g.iterrows():
                    lab = str(row[v["label_field"]]) if v["label_field"] else f"Unit {len(items) + 1}"
                    items.append((lab, row.geometry.area))
        if not items:
            return None, []
        items.sort(key=lambda x: _nat(x[0]))
        u = _area_unit(max(a for _, a in items))
        rows = [(l, _fmt_area(a, u)) for l, a in items]
        rows.append(("Total", _fmt_area(sum(a for _, a in items), u)))
        return "AREAS", rows
    if kind == "class_areas":
        areas = {}
        cls_r = next((r for r in rasters if r["role"] == "Classified"), None)
        if cls_r:
            counts, approx = cls_r["layer"].class_counts(crs, aoi)
            table = cls_r.get("_table") or _class_table(cls_r["layer"], counts.keys(), st)
            for v_, a in counts.items():
                if cls_r["layer"].nodata is not None and v_ == cls_r["layer"].nodata:
                    continue
                name = table.get(v_, {}).get("name", f"Class {v_}")
                areas[name] = areas.get(name, 0) + a
            if approx:
                warn.append("Class areas estimated from a resampled raster (very large image)")
        else:
            for i, v in enumerate(vectors):
                if v["role"] == "Thematic" and v["style_field"]:
                    g = gdfs[i]
                    if aoi is not None:
                        g = g.clip(aoi)
                    for k, a in g.geometry.area.groupby(g[v["style_field"]].astype(str)).sum().items():
                        areas[pretty(k)] = areas.get(pretty(k), 0) + a
        if not areas:
            return None, []
        tot = sum(areas.values()); u = "ha" if tot < 5e7 else "km2"
        for k, a in sorted(areas.items(), key=lambda x: -x[1]):
            rows.append((k, f"{_fmt_area(a, u)}  ({a / tot * 100:.1f}%)"))
        rows.append(("Total", _fmt_area(tot, u)))
        return "AREA BY CLASS" + (" (within boundary)" if aoi is not None else ""), rows
    if kind == "network_lengths":
        lens = {}
        for i, v in enumerate(vectors):
            if v["role"] in ("Road", "Drainage") and v["layer"].kind == "LineString":
                g = gdfs[i]
                key = g[v["style_field"]].astype(str) if v["style_field"] else None
                if key is None:
                    lens[v["role"]] = lens.get(v["role"], 0) + g.length.sum()
                else:
                    for k, l in g.length.groupby(key).sum().items():
                        lab = (f"Order {k}" if v["role"] == "Drainage" else pretty(k))
                        lens[lab] = lens.get(lab, 0) + l
        if not lens:
            return None, []
        tot = sum(lens.values()); u = "m" if tot < 2000 else "km"
        rows = [(k, _fmt_len(l, u)) for k, l in sorted(lens.items(), key=lambda x: -x[1])]
        rows.append(("Total", _fmt_len(tot, u)))
        return "LENGTH BY CLASS", rows
    if kind == "elevation":
        e = next((r for r in rasters if r["role"] == "Elevation"), None)
        if not e:
            return None, []
        b = aoi.bounds if aoi is not None else frame.bounds
        arr, mask, iext = e["layer"].read(crs, (b[0], b[2], b[1], b[3]), 1500)
        if arr is None:
            return None, []
        z = arr[0].astype("float64")
        if aoi is not None:
            from rasterio.features import geometry_mask
            from rasterio.transform import from_bounds as tfb
            tr = tfb(iext[0], iext[2], iext[1], iext[3], z.shape[1], z.shape[0])
            mask &= geometry_mask([aoi], out_shape=z.shape, transform=tr, invert=True)
        z = z[mask]; z = z[z > -1000]
        if not z.size:
            return None, []
        rows = [("Minimum", f"{z.min():,.1f} m"), ("Maximum", f"{z.max():,.1f} m"), ("Mean", f"{z.mean():,.1f} m"),
                ("Range", f"{z.max() - z.min():,.1f} m")]
        if plots and len(plots) == 1:
            rows.append(("Plot area", _fmt_area(rep["plots"][0]["area"], _area_unit(rep["plots"][0]["area"]))))
        return "ELEVATION" + (" (within boundary)" if aoi is not None else ""), rows
    return None, []


# =====================================================================================
# panel
ROW_MM = 5.4


def _legend_handles(ctx):
    S = ctx["S"]
    h = []
    if ctx["plots"]:
        lab = "Plot boundary" if ctx["T"]["id"] == "site_plan" else "Boundary"
        h.append(MPoly([[0, 0]], closed=True, fc=(1, 0.15, 0.15, 0.2), ec="#ff2020", lw=2 * S, label=lab))
        if ctx["T"]["boundary"].get("corners"):
            h.append(Line2D([], [], marker="o", ls="none", mfc="white", mec="#ff2020", mew=1.3 * S, ms=6 * S, label="Boundary corner"))
    seen = set()
    cbar = None
    for e in ctx["legend"]:
        if e["kind"] == "colorbar":
            cbar = e; continue
        key = (e["kind"], e["label"])
        if key in seen:
            continue
        seen.add(key)
        if e["kind"] == "patch":
            pe_ = [pe.withStroke(linewidth=3 * S, foreground=e["halo"])] if e.get("halo") else None
            fc = e["fc"] if e["fc"] == "none" else (*to_rgb(e["fc"]), e.get("alpha", 1.0))
            h.append(MPoly([[0, 0]], closed=True, fc=fc, ec=e["ec"], lw=e.get("lw", 0.8 * S), label=e["label"], path_effects=pe_))
        elif e["kind"] == "line":
            pe_ = [pe.withStroke(linewidth=e["lw"] + 2.2 * S, foreground=e["halo"])] if e.get("halo") else None
            h.append(Line2D([], [], color=e["color"], lw=e["lw"], ls="--" if e.get("dash") else "-", label=e["label"], path_effects=pe_))
        elif e["kind"] == "marker":
            h.append(Line2D([], [], marker="o", ls="none", mfc=e["color"], mec=e["ec"], ms=7 * S, label=e["label"]))
    if ctx["T"]["boundary"].get("side_lengths") and ctx["plots"]:
        h.append(Line2D([], [], ls="none", marker="s", ms=8 * S, mfc="white", mec="#ff2020", label="Side length (m)"))
    return h, cbar


def _notes_lines(ctx, w):
    chars = int(w / 1.62)
    out = []
    for n in ctx["notes"]:
        out += textwrap.wrap(n, chars, subsequent_indent="    ") or [""]
    return out


def _item_need(item, ctx, w):
    if item == "title":
        return 34
    if item == "scale":
        return 25
    if item == "notes":
        return 9.5 + 3.4 * len(_notes_lines(ctx, w))
    if item == "stats":
        if not ctx["rows"]:
            return 0
        return 11 + ROW_MM * min(len(ctx["rows"]), ctx.get("max_rows", 99))
    if item == "legend":
        h, cbar = ctx["handles"], ctx["cbar"]
        ncol = 1 if (len(h) <= 9 or w < 80) else 2
        rows = math.ceil(len(h) / ncol)
        return 12 + rows * 8 * 2.05 * 0.3528 + (16 if cbar else 0)
    if item == "locator":
        return 45
    return 0


def _draw_panel(ctx, columns, panel):
    ctx["handles"], ctx["cbar"] = _legend_handles(ctx)
    have_loc = _locator_possible(ctx)
    for (x0, y0, w, h), items in zip(columns, panel):
        items = [i for i in items if not (i == "locator" and not have_loc) and not (i == "stats" and not ctx["rows"])]
        if not items:
            continue
        gap = 3
        needs = {i: _item_need(i, ctx, w) for i in items}
        total = sum(needs.values()) + gap * (len(items) - 1)
        spare = h - total
        heights = dict(needs)
        flex = [i for i in ("locator", "legend") if i in items]
        if spare > 0 and flex:
            tgt = "locator" if "locator" in flex else flex[0]
            heights[tgt] += spare
        elif spare < 0:
            deficit = -spare
            if "stats" in items and len(ctx["rows"]) > 4:
                cut = min(len(ctx["rows"]) - 4, int(math.ceil(deficit / ROW_MM)))
                ctx["max_rows"] = len(ctx["rows"]) - cut
                heights["stats"] = _item_need("stats", ctx, w)
                deficit -= cut * ROW_MM
            if deficit > 0 and "locator" in items:
                red = min(deficit, heights["locator"] - 30); heights["locator"] -= red; deficit -= red
            if deficit > 0 and "legend" in items:
                heights["legend"] = max(20, heights["legend"] - deficit)
        y = y0 + h
        for i in items:
            hh = heights[i]
            y -= hh
            _DRAW[i](ctx, x0, y, w, hh)
            y -= gap


def _box(ctx, x0, y0, w, h, title=None):
    pg, S = ctx["pg"], ctx["S"]
    pg.rect(x0, y0, w, h, lw=0.7 * S, ec="#222")
    if title:
        pg.text(x0 + 3, y0 + h - 4.5, title, 8.5, fontweight="bold", color="#222", va="center")


def _p_title(ctx, x0, y0, w, h):
    pg, S = ctx["pg"], ctx["S"]
    _box(ctx, x0, y0, w, h)
    pg.rect(x0, y0 + h - 12, w, 12, fill=True, fc=NAVY, ec="#222", lw=0.7 * S)
    t = ctx["title"]
    fs = min(15.0, 15.0 * (w / 104) * 24 / max(len(t), 1))
    pg.text(x0 + w / 2, y0 + h - 6, t, fs, ha="center", va="center", fontweight="bold", color="white")
    sub = ctx["subtitle"]
    if sub:
        pg.text(x0 + w / 2, y0 + h - 18.5, sub, min(9.5, 9.5 * (w / 104) * 48 / max(len(sub), 1)),
                ha="center", va="center", color="#222")
    pg.text(x0 + w / 2, y0 + 6.5, f"Scale 1:{ctx['denom']:,} ({ctx['paper']} print)", 9, ha="center", va="center", color="#444")


def _locator_possible(ctx):
    if ctx["place_index"]:
        try:
            outer, inner, _, _ = ctx["place_index"].locator_layers(ctx["center_pt"], ctx["crs"])
            if outer is not None:
                ctx["_loc"] = ("place", outer, inner)
                return True
        except Exception:
            pass
    src = ctx["imagery"] or next((r["layer"] for r in ctx["rasters"]), None)
    if src is not None:
        ctx["_loc"] = ("raster", src)
        return True
    if ctx["vectors"]:
        ctx["_loc"] = ("vectors",)
        return True
    return False


def _p_locator(ctx, x0, y0, w, h):
    pg, S = ctx["pg"], ctx["S"]
    ext, gw, gh = ctx["ext"], ctx["gw"], ctx["gh"]
    mode = ctx["_loc"][0]
    _box(ctx, x0, y0, w, h, "LOCATION")
    iax = pg.axes(x0 + 3, y0 + 2, w - 6, h - 10)
    iax.set_xticks([]); iax.set_yticks([]); iax.set_aspect("equal")
    for sp in iax.spines.values():
        sp.set_linewidth(0.6 * S)
    _orig_plot = iax.plot
    rect = Rectangle((ext[0], ext[2]), gw, gh, fill=False, ec="black", lw=1 * S, ls="--", zorder=10)
    if mode == "place":
        _, outer, inner = ctx["_loc"]
        outer.plot(ax=iax, color="#eeeeee", edgecolor="#9a9a9a", linewidth=0.4 * S)
        if inner is not None:
            inner.plot(ax=iax, color="#fff2b3", edgecolor="#666", linewidth=0.7 * S)
            nm = ctx["places"]
        b = outer.total_bounds
        iax.set_xlim(b[0], b[2]); iax.set_ylim(b[1], b[3])
        # AOI: outline if visible, otherwise a marker
        if gw / (b[2] - b[0]) > 0.04:
            iax.add_patch(rect)
        p = ctx["center_pt"]
        iax.scatter([p.x], [p.y], s=40 * S * S, marker="s", c="#ff2020", edgecolors="white", linewidths=0.8 * S, zorder=11)
        lvl = ctx["places"]
        if lvl:
            first = next((lvl[k] for k in ("district", "block", "circle", "village", "town") if k in lvl), "")
            if first:
                pg.text(x0 + w - 3, y0 + h - 4.5, first, 7, ha="right", va="center", color="#555")
        iax.set_xlabel(""); iax.set_ylabel("")
        return
    if mode == "raster":
        rl = ctx["_loc"][1]
        b = transform_bounds(rl.crs.to_wkt(), ctx["crs"].to_wkt(), *rl.bounds) if rl.crs else rl.bounds
        e2 = (b[0], b[2], b[1], b[3])
        if rl.kind in ("rgb", "multiband"):
            bands = ctx["basemap"]["bands"] if ctx["basemap"] and ctx["basemap"]["layer"] is rl else [1, 2, 3]
            arr, mask, iext = rl.read(ctx["crs"], e2, 500 if ctx["preview"] else 1400, bands=bands)
            if arr is not None:
                rgb = _stretch(arr, mask)
                img = np.dstack([rgb[0], rgb[1], rgb[2], np.where(mask, 255, 0).astype(np.uint8)])
                iax.imshow(img, extent=iext, origin="upper")
        else:
            arr, mask, iext = rl.read(ctx["crs"], e2, 500)
            if arr is not None:
                z = np.ma.masked_array(arr[0].astype(float), ~mask)
                iax.imshow(z, extent=iext, origin="upper", cmap="Greys")
        iax.set_xlim(min(b[0], ext[0]), max(b[2], ext[1])); iax.set_ylim(min(b[1], ext[2]), max(b[3], ext[3]))
    else:
        allb = [ctx["gdfs"][i].total_bounds for i in range(len(ctx["vectors"]))]
        b = (min(x[0] for x in allb), min(x[1] for x in allb), max(x[2] for x in allb), max(x[3] for x in allb))
        for i, v in enumerate(ctx["vectors"]):
            ctx["gdfs"][i].plot(ax=iax, color="#bbbbbb", linewidth=0.4 * S, edgecolor="#888")
        iax.set_xlim(min(b[0], ext[0]), max(b[2], ext[1])); iax.set_ylim(min(b[1], ext[2]), max(b[3], ext[3]))
    if ctx["aoi"] is not None:
        a = ctx["aoi"]
        for poly in (a.geoms if hasattr(a, "geoms") else [a]):
            xs, ys = poly.exterior.xy
            iax.add_patch(MPoly(np.c_[xs, ys], closed=True, fc="#ff2020", ec="#ff2020", lw=1 * S, alpha=0.85, zorder=9))
    iax.add_patch(rect)
    iax.set_xlabel(""); iax.set_ylabel("")


def _p_legend(ctx, x0, y0, w, h):
    pg, S = ctx["pg"], ctx["S"]
    _box(ctx, x0, y0, w, h, "LEGEND")
    handles, cbar = ctx["handles"], ctx["cbar"]
    cb_h = 16 if cbar else 0
    if handles:
        lax = pg.axes(x0 + 2, y0 + 1 + cb_h, w - 4, h - 9 - cb_h); lax.axis("off")
        ncol = 1 if (len(handles) <= 9 or w < 80) else 2
        rows = math.ceil(len(handles) / ncol)
        need = rows * 8 * 2.05 * 0.3528 + 3
        f = min(1.0, (h - 9 - cb_h) / need) if need > 0 else 1.0
        fs = 8 * max(0.65, f) * (0.9 if ncol == 2 else 1)
        lax.legend(handles=handles, loc="upper left", frameon=False, fontsize=pg.fs(fs), handlelength=2.6,
                   labelspacing=0.75 * max(0.5, f), borderaxespad=0.2, ncol=ncol, columnspacing=1.0)
    if cbar:
        cax = pg.axes(x0 + 6, y0 + 7, w - 12, 3.2)
        grad = np.linspace(0, 1, 256)[None, :]
        cax.imshow(grad, aspect="auto", cmap=cbar["cmap"], extent=(cbar["vmin"], cbar["vmax"], 0, 1))
        cax.set_yticks([])
        cax.tick_params(labelsize=pg.fs(6.5), length=2 * S, width=0.5 * S, pad=1 * S)
        cax.set_xticks(np.linspace(cbar["vmin"], cbar["vmax"], 4))
        cax.set_xticklabels([f"{v:,.0f}" for v in np.linspace(cbar["vmin"], cbar["vmax"], 4)])
        pg.text(x0 + w / 2, y0 + 12.2, cbar["label"], 7, ha="center", va="center", color="#333")


def _p_stats(ctx, x0, y0, w, h):
    pg = ctx["pg"]
    _box(ctx, x0, y0, w, h, ctx["stats_title"])
    rows = ctx["rows"]
    mr = ctx.get("max_rows", len(rows))
    if mr < len(rows):
        keep = rows[:mr - 2] + [(f"+ {len(rows) - mr + 1} more", "")] + [rows[-1]]
        rows = keep
    yy = y0 + h - 11
    for k, v in rows:
        bold = k == "Total"
        lab = k if len(k) <= 30 else k[:28] + "…"
        pg.text(x0 + 4, yy, lab, 7.6, color="#111" if bold else "#444", va="center", fontweight="bold" if bold else "normal")
        pg.text(x0 + w - 4, yy, v, 8, fontweight="bold", color="#111", ha="right", va="center")
        yy -= ROW_MM


def _p_scale(ctx, x0, y0, w, h):
    pg, S, mpp = ctx["pg"], ctx["S"], ctx["mpp"]
    _box(ctx, x0, y0, w, h)
    na_x, na_y = x0 + 12, y0
    for sgn, fc in ((-1, "#111"), (1, "white")):
        pg.fig.add_artist(MPoly([[pg.x(na_x), pg.y(na_y + 18)], [pg.x(na_x + sgn * 4.5), pg.y(na_y + 3)],
                                 [pg.x(na_x), pg.y(na_y + 7)]], closed=True, transform=pg.fig.transFigure,
                                fc=fc, ec="#111", lw=0.8 * S))
    pg.text(na_x, na_y + 21.5, "N", 12, ha="center", va="center", fontweight="bold")
    avail = w - 36
    sb_len = _nice_le(avail * S * mpp, [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000,
                                         10000, 20000, 25000, 50000, 100000, 200000])
    sb_mm = sb_len / mpp / S
    segs = 1
    for n in ((5, 4, 2) if (str(int(sb_len))[0] in "15" or str(int(sb_len)).startswith("25")) else (4, 2)):
        if sb_mm / n >= 9:
            segs = n; break
    km = sb_len >= 1000
    sx0, sy0 = x0 + 26, y0 + 11
    for i in range(segs):
        pg.rect(sx0 + i * sb_mm / segs, sy0, sb_mm / segs, 2.2, fill=True, fc="#111" if i % 2 == 0 else "white", ec="#111", lw=0.7 * S)
    for i in range(segs + 1):
        v = i * sb_len / segs
        pg.text(sx0 + i * sb_mm / segs, sy0 + 4.4, f"{v / 1000:g}" if km else f"{v:g}", 7, ha="center")
    pg.text(sx0 + sb_mm + 2.5, sy0 + 1.1, "km" if km else "m", 7.5, va="center")
    pg.text(sx0 + sb_mm / 2, sy0 - 5, f"1:{ctx['denom']:,} at {ctx['paper']}", 8, ha="center", color="#333")


def _p_notes(ctx, x0, y0, w, h):
    pg = ctx["pg"]
    _box(ctx, x0, y0, w, h)
    yy = y0 + h - 4.5
    for ln in _notes_lines(ctx, w):
        pg.text(x0 + 3, yy, ln, 6.6, color="#333", va="center"); yy -= 3.4
    pb = ctx["prepared_by"]
    if pb:
        pg.text(x0 + 3, y0 + 2.8, f"Prepared by: {pb}", 7 if len(pb) < 48 * w / 104 else 6.0,
                fontweight="bold", color=NAVY, va="center")


_DRAW = {"title": _p_title, "locator": _p_locator, "legend": _p_legend, "stats": _p_stats,
         "scale": _p_scale, "notes": _p_notes}
