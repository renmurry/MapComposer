"""Build tiny synthetic datasets and render every built-in template - checks a packaged build works."""
import os
import sys
import tempfile
import traceback


def run(report_path=None):
    lines, ok = [], False
    try:
        import numpy as np
        import rasterio
        import geopandas as gpd
        from rasterio.transform import from_origin
        from shapely.geometry import box, LineString, Point
        from .detect import scan, context
        from .templates import load_templates, suggest
        from .compose import compose, Job
        from .place import PlaceIndex

        d = tempfile.mkdtemp(prefix="mapcomposer_selftest_")
        U = "EPSG:32646"; x0, y0 = 572000, 2866100
        tr = from_origin(x0, y0, 0.05, 0.05)
        yy, xx = np.mgrid[0:600, 0:600]
        rgb = np.stack([(xx % 255), (yy % 255), ((xx + yy) % 255), np.full_like(xx, 255)]).astype("uint8")
        with rasterio.open(os.path.join(d, "test_mosaic.tif"), "w", driver="GTiff", width=600, height=600, count=4,
                           dtype="uint8", crs=U, transform=tr) as dst:
            dst.write(rgb)
        dsm = (100 + xx * 0.01 + yy * 0.02).astype("float32")[None]
        with rasterio.open(os.path.join(d, "test_dsm.tif"), "w", driver="GTiff", width=600, height=600, count=1,
                           dtype="float32", crs=U, transform=tr, nodata=-10000) as dst:
            dst.write(dsm)
        cl = (1 + (xx // 200) + (yy // 300)).astype("uint8")[None]
        os.makedirs(os.path.join(d, "cls"))
        with rasterio.open(os.path.join(d, "cls", "lulc_class.tif"), "w", driver="GTiff", width=600, height=600, count=1,
                           dtype="uint8", crs=U, transform=tr, nodata=0) as dst:
            dst.write(cl)
        gpd.GeoDataFrame({"Id": [0]}, geometry=[box(x0 + 5, y0 - 25, x0 + 25, y0 - 5)], crs=U).to_file(os.path.join(d, "Boundary.shp"))
        gpd.GeoDataFrame({"type": ["AH_Road"]}, geometry=[LineString([(x0 + 1, y0 - 2), (x0 + 29, y0 - 2)])], crs=U).to_crs(4326).to_file(
            os.path.join(d, "Road.geojson"), driver="GeoJSON")
        ref = os.path.join(d, "ref"); os.makedirs(ref)
        gpd.GeoDataFrame({"DIST_NAME": ["TEST"]}, geometry=[box(x0 - 5000, y0 - 5000, x0 + 5000, y0 + 5000)], crs=U).to_file(
            os.path.join(ref, "districts.gpkg"), driver="GPKG")
        place = PlaceIndex(ref)
        lines.append(f"place levels: {[l for l, _, _ in place.levels]}")

        T = load_templates()
        s = scan(d)
        lines.append(f"scan: rasters={[(r['layer'].name, r['layer'].kind, r['role']) for r in s['rasters']]} "
                     f"vectors={[(v['layer'].name, v['role']) for v in s['vectors']]}")
        lines.append(f"suggested: {suggest(T, context(s))[0][1]['id']}")
        good = 0
        for t in T:
            job = Job(template=t, rasters=s["rasters"], vectors=s["vectors"], out_base=os.path.join(d, "st_" + t["id"]),
                      texts={"name": "Test"}, png_dpi=80, place_index=place)
            r = compose(job, log=lines.append)
            sizes = [os.path.getsize(o) for o in r["outputs"]]
            lines.append(f"{t['id']}: {[os.path.basename(o) for o in r['outputs']]} {sizes} warnings={r['warnings']}")
            good += all(x > 5000 for x in sizes)
        # themes + graduated colours + line labels
        from .styles import load_themes
        s2 = scan(d)
        for v in s2["vectors"]:
            if v["role"] == "Boundary":
                v["layer"].gdf["pop"] = [123]
        for tid in load_themes():
            r = compose(Job(template=next(t for t in T if t["id"] == "site_plan"), rasters=s["rasters"], vectors=s["vectors"],
                            out_base=os.path.join(d, "theme_" + tid), texts={"name": "x"}, theme=tid, png_dpi=60,
                            write_pdf=False), log=lambda m: None)
            lines.append(f"theme {tid}: {os.path.getsize(r['outputs'][0])} bytes")
            good += os.path.getsize(r["outputs"][0]) > 5000
        gp = os.path.join(d, "grad"); os.makedirs(gp)
        gpd.GeoDataFrame({"Ward": [f"W{i}" for i in range(12)], "Pop": [float(i * i) for i in range(12)]},
                         geometry=[box(x0 + i * 10, y0 - 40, x0 + i * 10 + 9, y0 - 10) for i in range(12)], crs=U).to_file(os.path.join(gp, "wards.shp"))
        gpd.GeoDataFrame({"name": ["Main Road"]}, geometry=[LineString([(x0, y0 - 45), (x0 + 120, y0 - 5)])], crs=U).to_file(os.path.join(gp, "roads.shp"))
        sg = scan(gp)
        for v in sg["vectors"]:
            if v["role"] == "Admin":
                v["style_field"] = "Pop"; v["classify"] = "jenks"
        r = compose(Job(template=next(t for t in T if t["id"] == "admin_map"), rasters=[], vectors=sg["vectors"],
                        out_base=os.path.join(d, "graduated"), texts={"name": "x"}, png_dpi=60, write_pdf=False), log=lambda m: None)
        lines.append(f"graduated: roles={[(v['layer'].name, v['role'], v['label_field']) for v in sg['vectors']]} ok")
        good += 1
        # which file formats this build can read
        import pyogrio
        with rasterio.Env() as env:
            rdr = env.drivers()
        vd = pyogrio.list_drivers()
        lines.append("vector drivers: " + ", ".join(k for k in ["OpenFileGDB", "DXF", "GPX", "MapInfo File", "GML", "FlatGeobuf",
                                                              "SQLite", "LIBKML", "KML", "GPKG", "ESRI Shapefile"] if k in vd))
        lines.append("raster drivers: " + ", ".join(k for k in ["GTiff", "AAIGrid", "EHdr", "SRTMHGT", "netCDF", "HDF5", "AIG",
                                                              "HFA", "SAGA", "PNG", "JPEG", "JP2OpenJPEG", "MBTiles"] if k in rdr))
        area = compose(Job(template=next(t for t in T if t["id"] == "site_plan"), rasters=s["rasters"], vectors=s["vectors"],
                           out_base=os.path.join(d, "area"), texts={"name": "x"}, write_png=False), log=lambda m: None)["plots"][0]["area"]
        lines.append(f"area: {area:.2f} (expected 400.00)")
        from .styles import load_themes as _lt
        ok = good == len(T) + len(_lt()) + 1 and abs(area - 400.0) < 0.01 and len(T) >= 5
    except Exception:
        lines.append(traceback.format_exc())
    lines.append("SELFTEST " + ("PASSED" if ok else "FAILED"))
    text = "\n".join(lines)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(text)
    elif sys.stdout:
        print(text)
    return 0 if ok else 1
