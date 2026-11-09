"""MapComposer desktop app (PySide6)."""
import os
import sys
import tempfile
import traceback
from dataclasses import replace

from PySide6.QtCore import Qt, QThread, Signal, QSettings, QUrl
from PySide6.QtGui import QPixmap, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QTabWidget, QPlainTextEdit, QSplitter, QGroupBox, QScrollArea, QDialog,
    QProgressBar, QAbstractItemView,
)

from . import __version__
from .detect import scan, context, VROLES, RROLES
from .templates import load_templates, suggest
from .compose import compose, Job, PAPERS
from .place import PlaceIndex
from .paths import ensure_user_copies, user_dir
from .batch import run_batch, plan, safe_name

SCALES = ["Auto", "1:100", "1:200", "1:250", "1:500", "1:1000", "1:2000", "1:2500", "1:5000", "1:10000",
          "1:25000", "1:50000", "1:100000", "1:250000"]
POS = [("Drone GNSS only (no GCP / RTK)", "gnss"), ("RTK / PPK", "rtk"), ("Ground control points (GCP)", "gcp"),
       ("Not applicable", "none")]


class Worker(QThread):
    log = Signal(str); done = Signal(object); failed = Signal(str)

    def __init__(self, fn):
        super().__init__(); self.fn = fn; self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.done.emit(self.fn(self.log.emit, lambda: self._stop))
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc(limit=4)}")


class PreviewDialog(QDialog):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview"); self.resize(1300, 940)
        lay = QVBoxLayout(self); sa = QScrollArea(); sa.setWidgetResizable(True)
        lbl = QLabel(); lbl.setAlignment(Qt.AlignCenter)
        lbl.setPixmap(QPixmap(path).scaledToWidth(1260, Qt.SmoothTransformation))
        sa.setWidget(lbl); lay.addWidget(sa)


def _combo(items, current=""):
    c = QComboBox(); c.addItems(items)
    if current in items:
        c.setCurrentText(current)
    return c


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"MapComposer {__version__}")
        self.resize(1460, 900)
        self.settings = QSettings("MapComposer", "MapComposer")
        try:
            ensure_user_copies()
        except Exception:
            pass
        self.templates = load_templates()
        self.sc = None
        self.place = None
        self.worker = None

        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        split = QSplitter(Qt.Horizontal); root.addWidget(split, 1)
        self.tabs = QTabWidget(); split.addWidget(self.tabs)
        self.tabs.addTab(self._single_tab(), "Single map")
        self.tabs.addTab(self._batch_tab(), "Batch (many folders)")
        self.tabs.currentChanged.connect(self._update_buttons)
        sa = QScrollArea(); sa.setWidgetResizable(True); sa.setMinimumWidth(450)
        sa.setWidget(self._settings_panel()); split.addWidget(sa)
        split.setSizes([980, 470])

        btns = QHBoxLayout()
        self.btn_preview = QPushButton("Preview"); self.btn_preview.clicked.connect(self.preview)
        self.btn_go = QPushButton("Generate map"); self.btn_go.clicked.connect(self.generate)
        self.btn_go.setStyleSheet("font-weight:bold; padding:6px 18px;")
        self.btn_stop = QPushButton("Stop batch"); self.btn_stop.clicked.connect(self._stop); self.btn_stop.setEnabled(False)
        self.btn_open = QPushButton("Open output folder"); self.btn_open.clicked.connect(self.open_output)
        self.btn_tpl = QPushButton("Edit templates / styles..."); self.btn_tpl.clicked.connect(self.open_templates)
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.setVisible(False); self.progress.setMaximumWidth(180)
        for b in (self.btn_preview, self.btn_go, self.btn_stop, self.btn_open, self.btn_tpl):
            btns.addWidget(b)
        btns.addWidget(self.progress); btns.addStretch(1)
        root.addLayout(btns)
        self.logbox = QPlainTextEdit(); self.logbox.setReadOnly(True); self.logbox.setMaximumHeight(160)
        self.logbox.setFont(QFont("Consolas", 9)); root.addWidget(self.logbox)
        self._fill_template_combos()
        self._load_settings()
        self._update_buttons()

    # ------------------------------------------------------------------ tabs
    def _single_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.folder_edit = QLineEdit(); self.folder_edit.setPlaceholderText("Folder with your data (searched including sub-folders)")
        b1 = QPushButton("Browse..."); b1.clicked.connect(self.browse_folder)
        b2 = QPushButton("Add files..."); b2.clicked.connect(self.add_files)
        b3 = QPushButton("Scan"); b3.clicked.connect(lambda: self.do_scan())
        row.addWidget(QLabel("Data folder:")); row.addWidget(self.folder_edit, 1)
        for b in (b1, b2, b3):
            row.addWidget(b)
        lay.addLayout(row)
        self.extra_files = []

        trow = QHBoxLayout()
        self.tpl_combo = QComboBox(); self.tpl_combo.currentIndexChanged.connect(self._template_changed)
        self.tpl_hint = QLabel(""); self.tpl_hint.setStyleSheet("color:#555;")
        trow.addWidget(QLabel("Map type:")); trow.addWidget(self.tpl_combo, 1); trow.addWidget(self.tpl_hint, 2)
        lay.addLayout(trow)

        lay.addWidget(QLabel("Rasters - Role decides how each is drawn. Bands: which bands show as red,green,blue (e.g. 3,2,1 for false colour)."))
        self.rtable = QTableWidget(0, 5)
        self.rtable.setHorizontalHeaderLabels(["Raster", "Detected", "Cell size", "Role", "Bands"])
        self._table_setup(self.rtable); self.rtable.setMaximumHeight(150)
        lay.addWidget(self.rtable)
        lay.addWidget(QLabel("Vector layers - check each Role. 'Style by' colours features by a field; 'Label' writes a field on the map."))
        self.vtable = QTableWidget(0, 6)
        self.vtable.setHorizontalHeaderLabels(["Layer", "Geometry", "Features", "Role", "Style by", "Label"])
        self._table_setup(self.vtable)
        lay.addWidget(self.vtable, 1)
        r2 = QHBoxLayout()
        self.out_edit = QLineEdit(); self.out_edit.setPlaceholderText("Same as data folder")
        bo = QPushButton("..."); bo.setMaximumWidth(32); bo.clicked.connect(self._pick_out)
        r2.addWidget(QLabel("Save to:")); r2.addWidget(self.out_edit, 1); r2.addWidget(bo)
        lay.addLayout(r2)
        return w

    def _table_setup(self, t):
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, t.columnCount()):
            t.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        t.verticalHeader().setVisible(False); t.setSelectionMode(QAbstractItemView.NoSelection)

    def _batch_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.parent_edit = QLineEdit(); self.parent_edit.setPlaceholderText("Parent folder with one sub-folder per map")
        b = QPushButton("Browse..."); b.clicked.connect(self.browse_parent)
        b2 = QPushButton("Check folders"); b2.clicked.connect(self.check_batch)
        row.addWidget(QLabel("Parent folder:")); row.addWidget(self.parent_edit, 1); row.addWidget(b); row.addWidget(b2)
        lay.addLayout(row)
        trow = QHBoxLayout()
        self.btpl_combo = QComboBox()
        trow.addWidget(QLabel("Map type:")); trow.addWidget(self.btpl_combo, 1)
        lay.addLayout(trow)
        lay.addWidget(QLabel("Each sub-folder becomes one map saved inside it. Titles come from each template "
                             "({name} = folder name); text fields on the right (date, source, prepared by...) are shared."))
        self.btable = QTableWidget(0, 3)
        self.btable.setHorizontalHeaderLabels(["Folder", "Map type", "Status"])
        self._table_setup(self.btable); self.btable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.btable, 1)
        return w

    def _settings_panel(self):
        w = QWidget(); lay = QVBoxLayout(w)
        g = QGroupBox("Map text  ({name} {place_line} {district} {village} {ward} ... are filled in)"); f = QFormLayout(g)
        self.title_edit = QLineEdit(); self.name_edit = QLineEdit(); self.sub_edit = QLineEdit()
        self.date_edit = QLineEdit(); self.date_edit.setPlaceholderText("Flight / image date, e.g. 24 Oct 2026")
        self.src_edit = QLineEdit(); self.src_edit.setPlaceholderText("e.g. SISDP 2.0, LISS-IV 2025, field survey")
        self.pos_combo = QComboBox(); self.pos_combo.addItems([p[0] for p in POS])
        self.rmse_edit = QLineEdit(); self.rmse_edit.setPlaceholderText("e.g. 0.04 m")
        self.pos_combo.currentIndexChanged.connect(lambda i: self.rmse_edit.setEnabled(POS[i][1] == "gcp"))
        self.prep_edit = QLineEdit()
        self.disc_check = QCheckBox("Disclaimer"); self.disc_edit = QLineEdit()
        self.disc_check.toggled.connect(self.disc_edit.setEnabled)
        for lab, wd in (("Title:", self.title_edit), ("Name:", self.name_edit), ("Subtitle:", self.sub_edit),
                        ("Date:", self.date_edit), ("Data source:", self.src_edit), ("Positioning:", self.pos_combo),
                        ("GCP RMSE:", self.rmse_edit), ("Prepared by:", self.prep_edit)):
            f.addRow(lab, wd)
        f.addRow(self.disc_check, self.disc_edit)
        lay.addWidget(g)

        g2 = QGroupBox("Layout and options"); f2 = QFormLayout(g2)
        self.paper_combo = _combo(list(PAPERS.keys()), "A3")
        self.orient_combo = _combo(["Template default", "Landscape", "Portrait"])
        self.scale_combo = QComboBox(); self.scale_combo.setEditable(True); self.scale_combo.addItems(SCALES)
        self.sides_check = QCheckBox("Label boundary side lengths")
        self.str_check = QCheckBox("Straighten near-straight sides")
        self.str_tol = QDoubleSpinBox(); self.str_tol.setRange(0.01, 5); self.str_tol.setDecimals(2); self.str_tol.setSuffix(" m")
        self.str_check.toggled.connect(self.str_tol.setEnabled)
        self.cont_check = QCheckBox("Contours from elevation raster")
        self.cont_iv = QDoubleSpinBox(); self.cont_iv.setRange(0.25, 500); self.cont_iv.setSuffix(" m interval")
        self.cont_check.toggled.connect(self.cont_iv.setEnabled)
        self.pdf_check = QCheckBox("PDF"); self.png_check = QCheckBox("PNG")
        self.dpi_spin = QSpinBox(); self.dpi_spin.setRange(100, 600); self.dpi_spin.setSingleStep(50); self.dpi_spin.setSuffix(" dpi")
        orow = QHBoxLayout(); orow.addWidget(self.pdf_check); orow.addWidget(self.png_check); orow.addWidget(self.dpi_spin); orow.addStretch(1)
        f2.addRow("Paper:", self.paper_combo); f2.addRow("Orientation:", self.orient_combo); f2.addRow("Scale:", self.scale_combo)
        f2.addRow(self.sides_check); f2.addRow(self.str_check, self.str_tol); f2.addRow(self.cont_check, self.cont_iv)
        f2.addRow("Save as:", orow)
        lay.addWidget(g2)

        g3 = QGroupBox("Place awareness (optional)"); v3 = QVBoxLayout(g3)
        r = QHBoxLayout()
        self.ref_edit = QLineEdit(); self.ref_edit.setPlaceholderText("Folder of state / district / block / village / ward boundaries")
        br = QPushButton("..."); br.setMaximumWidth(32); br.clicked.connect(self._pick_ref)
        r.addWidget(self.ref_edit, 1); r.addWidget(br); v3.addLayout(r)
        self.ref_status = QLabel("Not set - locator uses the imagery instead."); self.ref_status.setWordWrap(True)
        self.ref_status.setStyleSheet("color:#555;"); v3.addWidget(self.ref_status)
        lay.addWidget(g3)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------ settings
    def _load_settings(self):
        s = self.settings; b = lambda k, d: s.value(k, d) in (True, "true")
        self.prep_edit.setText(s.value("prepared_by", "Nagaland GIS & Remote Sensing Centre, Kohima"))
        self.pos_combo.setCurrentIndex(int(s.value("pos", 0))); self.rmse_edit.setEnabled(POS[self.pos_combo.currentIndex()][1] == "gcp")
        self.src_edit.setText(s.value("source", ""))
        self.disc_edit.setText(s.value("disclaimer", "Boundaries digitised from imagery. Not a cadastral survey."))
        self.disc_check.setChecked(b("disc_on", False)); self.disc_edit.setEnabled(self.disc_check.isChecked())
        self.paper_combo.setCurrentText(s.value("paper", "A3")); self.scale_combo.setCurrentText(s.value("scale", "Auto"))
        self.str_check.setChecked(b("straighten", False)); self.str_tol.setValue(float(s.value("str_tol", 0.15)))
        self.str_tol.setEnabled(self.str_check.isChecked())
        self.pdf_check.setChecked(b("pdf", True)); self.png_check.setChecked(b("png", True))
        self.dpi_spin.setValue(int(s.value("dpi", 300)))
        self.folder_edit.setText(s.value("last_folder", "")); self.parent_edit.setText(s.value("last_parent", ""))
        self.ref_edit.setText(s.value("ref_folder", ""))
        if self.ref_edit.text():
            self._load_ref(self.ref_edit.text())

    def _save_settings(self):
        s = self.settings
        for k, v in (("prepared_by", self.prep_edit.text()), ("pos", self.pos_combo.currentIndex()),
                     ("source", self.src_edit.text()), ("disclaimer", self.disc_edit.text()),
                     ("disc_on", self.disc_check.isChecked()), ("paper", self.paper_combo.currentText()),
                     ("scale", self.scale_combo.currentText()), ("straighten", self.str_check.isChecked()),
                     ("str_tol", self.str_tol.value()), ("pdf", self.pdf_check.isChecked()),
                     ("png", self.png_check.isChecked()), ("dpi", self.dpi_spin.value()),
                     ("last_folder", self.folder_edit.text()), ("last_parent", self.parent_edit.text()),
                     ("ref_folder", self.ref_edit.text())):
            s.setValue(k, v)

    def closeEvent(self, ev):
        self._save_settings(); super().closeEvent(ev)

    # ------------------------------------------------------------------ helpers
    def log(self, m):
        self.logbox.appendPlainText(m)
        self.logbox.verticalScrollBar().setValue(self.logbox.verticalScrollBar().maximum())

    def _update_buttons(self, *_):
        single = self.tabs.currentIndex() == 0
        self.btn_preview.setEnabled(single); self.btn_go.setText("Generate map" if single else "Run batch")

    def _busy(self, on):
        self.progress.setVisible(on)
        for b in (self.btn_preview, self.btn_go):
            b.setEnabled(not on)
        self.btn_stop.setEnabled(on and self.tabs.currentIndex() == 1)
        if not on:
            self._update_buttons()

    def _fill_template_combos(self, ranked=None):
        self.tpl_combo.blockSignals(True)
        cur = self.tpl_combo.currentData()
        self.tpl_combo.clear()
        order = [t for _, t in ranked] if ranked else self.templates
        best = ranked[0][1]["id"] if ranked and ranked[0][0] > 0 else None
        for t in order:
            self.tpl_combo.addItem(("★ " if t["id"] == best else "") + t["name"], t["id"])
        if best:
            self.tpl_combo.setCurrentIndex(0)
        elif cur:
            i = self.tpl_combo.findData(cur); self.tpl_combo.setCurrentIndex(max(0, i))
        self.tpl_combo.blockSignals(False)
        self._template_changed()
        self.btpl_combo.clear(); self.btpl_combo.addItem("Auto - suggested per folder", "")
        for t in self.templates:
            self.btpl_combo.addItem(t["name"], t["id"])

    def _template(self):
        tid = self.tpl_combo.currentData()
        return next((t for t in self.templates if t["id"] == tid), self.templates[0])

    def _template_changed(self, *_):
        t = self._template()
        self.title_edit.setText(t["text"]["title"]); self.sub_edit.setText(t["text"]["subtitle"])
        self.sides_check.setChecked(bool(t["boundary"].get("side_lengths")))
        self.cont_check.setChecked(bool(t["contours"]["default"])); self.cont_iv.setValue(float(t["contours"]["interval"]))
        self.cont_iv.setEnabled(self.cont_check.isChecked())
        self.tpl_hint.setText(t.get("description", ""))

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Save maps to", self.out_edit.text() or self.folder_edit.text())
        if d:
            self.out_edit.setText(d)

    def _pick_ref(self):
        d = QFileDialog.getExistingDirectory(self, "Reference boundaries folder", self.ref_edit.text())
        if d:
            self.ref_edit.setText(d); self._load_ref(d)

    def _load_ref(self, d):
        try:
            self.place = PlaceIndex(d)
        except Exception as e:
            self.place = None; self.ref_status.setText(f"Could not read: {e}"); return
        if self.place:
            self.ref_status.setText("Loaded: " + ", ".join(f"{l} ({lyr.name}, name field '{nf}')" for l, lyr, nf in self.place.levels))
        else:
            self.ref_status.setText("No usable layers found. Name files by level, e.g. districts.shp, villages.gpkg, wards.shp.")
        for w in self.place.warnings if self.place else []:
            self.log("  reference: " + w)

    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Data folder", self.folder_edit.text())
        if d:
            self.folder_edit.setText(d); self.extra_files = []; self.do_scan()

    def add_files(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Add data files", self.folder_edit.text(),
                                             "GIS data (*.tif *.tiff *.img *.jp2 *.shp *.gpkg *.kml *.kmz *.geojson *.json *.csv)")
        if fs:
            self.extra_files += fs; self.do_scan()

    def browse_parent(self):
        d = QFileDialog.getExistingDirectory(self, "Parent folder", self.parent_edit.text())
        if d:
            self.parent_edit.setText(d); self.check_batch()

    def open_output(self):
        d = (self.out_edit.text() or self.folder_edit.text()) if self.tabs.currentIndex() == 0 else self.parent_edit.text()
        if d and os.path.isdir(d):
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    def open_templates(self):
        d = ensure_user_copies()
        QMessageBox.information(self, "Templates and styles",
                                f"Your editable copies are in:\n{d}\n\n- templates\\*.yaml : one file per map type. Copy one, "
                                "change 'id' and 'name', and it appears in the list after restarting.\n- styles.yaml : colours "
                                "and recognised class names.\n\nDelete a file to get the built-in version back.")
        QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    # ------------------------------------------------------------------ scanning
    def do_scan(self):
        folder = self.folder_edit.text().strip().strip('"')
        if folder and not os.path.isdir(folder):
            QMessageBox.warning(self, "MapComposer", "That folder does not exist."); return
        if not folder and not self.extra_files:
            QMessageBox.warning(self, "MapComposer", "Choose a data folder or add files."); return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.sc = scan(folder, self.extra_files)
        except Exception as e:
            QApplication.restoreOverrideCursor(); QMessageBox.critical(self, "Scan failed", str(e)); return
        QApplication.restoreOverrideCursor()
        sc = self.sc
        base = folder or os.path.dirname(self.extra_files[0])
        rel = lambda p: os.path.relpath(p, base) if p.startswith(base) else p
        self.rtable.setRowCount(0)
        for r in sc["rasters"]:
            i = self.rtable.rowCount(); self.rtable.insertRow(i); rl = r["layer"]
            res = f"{rl.res_m * 100:.1f} cm" if rl.res_m < 1 else f"{rl.res_m:g} m"
            for c, v in enumerate([rel(rl.path), f"{rl.kind} ({rl.count} band{'s' if rl.count > 1 else ''})", res]):
                it = QTableWidgetItem(v); it.setFlags(it.flags() & ~Qt.ItemIsEditable); self.rtable.setItem(i, c, it)
            self.rtable.setCellWidget(i, 3, _combo(RROLES, r["role"]))
            be = QLineEdit(",".join(map(str, r["bands"]))); be.setMaximumWidth(70)
            be.setEnabled(rl.count >= 3); self.rtable.setCellWidget(i, 4, be)
        self.vtable.setRowCount(0)
        for v in sc["vectors"]:
            i = self.vtable.rowCount(); self.vtable.insertRow(i); lyr = v["layer"]
            for c, val in enumerate([rel(lyr.path) + (f" [{lyr.sublayer}]" if lyr.sublayer else ""), lyr.kind, str(len(lyr))]):
                it = QTableWidgetItem(val); it.setFlags(it.flags() & ~Qt.ItemIsEditable); self.vtable.setItem(i, c, it)
            self.vtable.setCellWidget(i, 3, _combo(VROLES, v["role"]))
            self.vtable.setCellWidget(i, 4, _combo([""] + lyr.string_fields(), v["style_field"]))
            self.vtable.setCellWidget(i, 5, _combo([""] + lyr.label_fields(), v["label_field"]))
            self.vtable.cellWidget(i, 3).currentTextChanged.connect(self._roles_changed)
        for c in range(self.rtable.rowCount()):
            self.rtable.cellWidget(c, 3).currentTextChanged.connect(self._roles_changed)
        self.name_edit.setText(sc["name"])
        self._roles_changed()
        self.log(f"Scanned: {len(sc['rasters'])} raster(s), {len(sc['vectors'])} vector layer(s)")
        for w in sc["warnings"]:
            self.log("  note: " + w)

    def _apply_tables(self):
        for i, r in enumerate(self.sc["rasters"]):
            r["role"] = self.rtable.cellWidget(i, 3).currentText()
            try:
                r["bands"] = [int(x) for x in self.rtable.cellWidget(i, 4).text().replace(" ", "").split(",") if x]
            except ValueError:
                raise ValueError(f"Bands for {r['layer'].name} must be numbers like 3,2,1")
            if r["role"] == "Basemap":
                if len(r["bands"]) == 1:
                    r["bands"] = r["bands"] * 3
                if len(r["bands"]) != 3 or any(b < 1 or b > r["layer"].count for b in r["bands"]):
                    raise ValueError(f"{r['layer'].name}: choose 3 bands between 1 and {r['layer'].count}")
        for i, v in enumerate(self.sc["vectors"]):
            v["role"] = self.vtable.cellWidget(i, 3).currentText()
            v["style_field"] = self.vtable.cellWidget(i, 4).currentText()
            v["label_field"] = self.vtable.cellWidget(i, 5).currentText()

    def _roles_changed(self, *_):
        if not self.sc:
            return
        try:
            self._apply_tables()
        except ValueError:
            pass
        ranked = suggest(self.templates, context(self.sc))
        self._fill_template_combos(ranked)
        best = ranked[0]
        self.log(f"  suggested map type: {best[1]['name']}" if best[0] > 0 else "  no map type stands out - pick one")

    def check_batch(self):
        parent = self.parent_edit.text().strip().strip('"')
        if not os.path.isdir(parent):
            QMessageBox.warning(self, "MapComposer", "Choose an existing parent folder."); return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            p = plan(parent, self.templates, self.btpl_combo.currentData() or None)
        finally:
            QApplication.restoreOverrideCursor()
        self.btable.setRowCount(0)
        for f, s, t, reason in p:
            i = self.btable.rowCount(); self.btable.insertRow(i)
            for c, v in enumerate([os.path.basename(f), t["name"] if t else "-", reason]):
                self.btable.setItem(i, c, QTableWidgetItem(v))
        self.log(f"Batch check: {sum(1 for x in p if x[2])} of {len(p)} folder(s) ready")

    # ------------------------------------------------------------------ run
    def _job_base(self):
        txt = self.scale_combo.currentText().strip().lower().replace(",", "")
        scale = 0
        if txt and txt != "auto":
            try:
                scale = int(float(txt.split(":")[-1]))
            except ValueError:
                raise ValueError("Scale must look like 1:500 or Auto")
        if not (self.pdf_check.isChecked() or self.png_check.isChecked()):
            raise ValueError("Tick PDF and/or PNG.")
        orient = {"Landscape": "landscape", "Portrait": "portrait"}.get(self.orient_combo.currentText(), "")
        texts = dict(name=self.name_edit.text().strip(), title=self.title_edit.text().strip(),
                     subtitle=self.sub_edit.text().strip(), date=self.date_edit.text().strip(),
                     source=self.src_edit.text().strip(), positioning=POS[self.pos_combo.currentIndex()][1],
                     rmse=self.rmse_edit.text().strip(), prepared_by=self.prep_edit.text().strip(),
                     disclaimer=self.disc_edit.text().strip() if self.disc_check.isChecked() else "")
        return Job(template=self._template(), rasters=[], vectors=[], out_base="", texts=texts,
                   paper=self.paper_combo.currentText(), orientation=orient, scale=scale,
                   side_lengths=self.sides_check.isChecked(), straighten=self.str_check.isChecked(),
                   straighten_tol=self.str_tol.value(), contours=self.cont_check.isChecked(),
                   contour_interval=self.cont_iv.value(), png_dpi=self.dpi_spin.value(),
                   write_pdf=self.pdf_check.isChecked(), write_png=self.png_check.isChecked(), place_index=self.place)

    def _single_job(self):
        if not self.sc:
            raise ValueError("Choose a data folder and Scan first.")
        self._apply_tables()
        job = self._job_base()
        t = job.template
        out_dir = self.out_edit.text().strip() or self.sc["folder"] or os.path.dirname(self.extra_files[0])
        os.makedirs(out_dir, exist_ok=True)
        name = job.texts["name"] or "map"
        return replace(job, rasters=self.sc["rasters"], vectors=self.sc["vectors"],
                       out_base=os.path.join(out_dir, f"{safe_name(name + '_' + t['id'])}_{job.paper}"))

    def _start(self, fn, on_done):
        self._save_settings()
        self.worker = Worker(fn)
        self.worker.log.connect(self.log)
        self.worker.done.connect(lambda r: (self._busy(False), on_done(r)))
        self.worker.failed.connect(lambda m: (self._busy(False), self.log("ERROR: " + m.splitlines()[0]),
                                              QMessageBox.critical(self, "MapComposer", m)))
        self._busy(True); self.worker.start()

    def preview(self):
        try:
            job = self._single_job()
        except Exception as e:
            QMessageBox.warning(self, "MapComposer", str(e)); return
        job = replace(job, out_base=os.path.join(tempfile.gettempdir(), "mapcomposer"))
        self.log("Rendering preview...")
        self._start(lambda log, stop: compose(job, preview=True, log=log),
                    lambda r: (self._report(r), PreviewDialog(r["outputs"][0], self).exec()))

    def generate(self):
        if self.tabs.currentIndex() == 1:
            return self.batch()
        try:
            job = self._single_job()
        except Exception as e:
            QMessageBox.warning(self, "MapComposer", str(e)); return
        self.log(f"Generating {job.template['name']} ...")
        self._start(lambda log, stop: compose(job, log=log), self._report)

    def batch(self):
        parent = self.parent_edit.text().strip().strip('"')
        if not os.path.isdir(parent):
            QMessageBox.warning(self, "MapComposer", "Choose an existing parent folder."); return
        try:
            base = self._job_base()
        except Exception as e:
            QMessageBox.warning(self, "MapComposer", str(e)); return
        texts = dict(base.texts); texts["title"] = ""; texts["subtitle"] = None
        base = replace(base, texts=texts, orientation="" if self.orient_combo.currentIndex() == 0 else base.orientation,
                       side_lengths=None, contours=None, contour_interval=0)
        fixed = self.btpl_combo.currentData() or None
        self.log(f"Batch: {parent}")

        def done(res):
            ok = sum(1 for r in res if r[1] == "ok")
            self.log(f"Batch finished: {ok} map(s) made, {len(res) - ok} skipped or failed")
        self._start(lambda log, stop: run_batch(parent, self.templates, base, fixed, log=log, stop_flag=stop), done)

    def _stop(self):
        if self.worker:
            self.worker.stop(); self.log("Stopping after the current folder...")

    def _report(self, r):
        for w in r["warnings"]:
            self.log("  note: " + w)
        if r.get("places"):
            self.log("  place: " + ", ".join(f"{k}={v}" for k, v in r["places"].items()))
        for p in r["plots"]:
            self.log(f"  {p['label']}: {p['area']:,.2f} m² / {p['sqft']:,.2f} sq ft, perimeter {p['perimeter']:,.2f} m")
        self.log(f"  {r.get('title', '')} | scale 1:{r['scale']:,} | {r['crs']}")
        for o in r["outputs"]:
            if not o.endswith("_preview.png"):
                self.log("  saved: " + o)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MapComposer"); app.setOrganizationName("MapComposer")
    w = MainWindow(); w.show()
    sys.exit(app.exec())
