# MapComposer

Internal Windows tool for composing print-ready maps (PDF / PNG) from rasters and vector layers.

## Map types (templates)

| Template | For | Panel statistics |
|---|---|---|
| Site plan | Plot boundary on a drone ortho | Area (m², sq ft), perimeter, centroid; side lengths on the map |
| Admin / ward map | Wards, villages, districts | Area per unit, labels on the map |
| Thematic | LULC polygons or a classified raster | Area by class (ha, %) |
| Network | Roads and/or drainage | Length by class / stream order |
| Terrain | DSM / DEM | Min / max / mean elevation, hillshade, contours |

Templates are YAML files. **Edit templates / styles…** opens your editable copies
(`%APPDATA%\MapComposer`). Copy a template, change its `id` and `name`, and it appears as a new map type.
`styles.yaml` holds colours and the class names that are recognised automatically (road types, LULC classes,
stream order fields).

## Themes

Every map can be drawn in five themes: **Classic** (navy title bar), **Atlas** (cream paper, serif),
**Night** (dark, teal accents), **Minimal** (open white layout, one accent colour) and **Blueprint**
(engineering title block, monospace). **Compare themes** renders the current map in all of them side by side;
**Also export every theme** writes one file per theme. Fonts are bundled (SIL Open Font License, see
`mapcomposer/resources/fonts`) so output looks the same on every PC. Themes live in `themes.yaml`; copy a
block to make your own.

## Styling

- **Style by** a text field gives categories; a numeric field (marked #) gives **graduated colours**
  (natural breaks, quantile or equal interval, 3-9 classes) for polygons, lines and points.
- **Label** writes a field on the map: inside polygons, beside points, and **along lines** for road and
  stream names (repeated on long lines, not crowded).
- **Layer order**: move rows with ▲▼; tick *Draw layers in table order* (top row is drawn on top).
- Several ortho tiles in one folder are joined into one seamless image.

## Context awareness

- **Scan** reads the folder (and sub-folders): rasters are classed as RGB ortho, multiband satellite,
  classified, elevation or continuous; vector layers get a role (Boundary, Admin, Thematic, Road, Drainage,
  Building, Points, Contour, Other) from their name, geometry and fields.
- The map type is suggested from what was found (★ in the list). Every role and the map type can be changed.
- Recognised fields are styled automatically: road type (NH, SH, AH, district, village…), LULC class names,
  stream order.
- Data in lat/long (KML, GeoJSON, CSV) is placed in the correct UTM zone automatically.
- **Place awareness**: point *Place awareness* at a folder of reference boundaries named by level
  (`state…`, `district…`, `block…`, `circle…`, `village…`, `ward…`). The map centre is looked up to fill
  `{district}`, `{village}`, `{ward}`, `{place_line}` in titles, and the locator shows the district map.

## Inputs

- Rasters: GeoTIFF, ERDAS IMG, JPEG2000, VRT, ASCII grid (.asc), BIL/BIP/BSQ, SRTM .hgt, USGS .dem, ER Mapper,
  NetCDF / HDF5 (each variable listed separately), SAGA, IDRISI, MBTiles, ESRI grid folders, and PNG/JPG with a
  world file. Any CRS, reprojected on the fly; choose display bands for multiband imagery.
- Vectors: Shapefile, File geodatabase (.gdb), GeoPackage, KML/KMZ, GeoJSON, DXF, GPX, MapInfo TAB/MIF, GML,
  FlatGeobuf, SpatiaLite, CSV points (lat/long or E/N columns).
- Not supported (proprietary, not in open GDAL builds): DWG, ECW, MrSID.
A classified raster can carry a colour table, or a sidecar CSV with the same name (`value,name[,#colour]`).

## Using it

1. Download `MapComposer-windows.zip` from the latest Actions run (Artifacts) or Releases, unzip, run `MapComposer.exe`.
2. **Single map**: Browse to a data folder → check roles → fill text on the right → **Preview** → **Generate map**.
3. **Batch**: choose a parent folder → **Check folders** → **Run batch**. Each sub-folder gets a map with
   its suggested (or a fixed) map type, saved inside that sub-folder.

Text, options and the reference folder are remembered between sessions.

## Building

Each push to `main` builds on GitHub Actions (Windows), runs `MapComposer.exe --selftest` (renders every
template on synthetic data) and uploads the zip. A tag like `v1.0.0` also attaches it to a release.

```
pip install -r requirements.txt
python MapComposer.py
python MapComposer.py --selftest report.txt
```
