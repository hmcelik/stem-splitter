"""Stem Splitter - desktop UI (PySide6)."""
from __future__ import annotations

import logging
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QMimeData, QObject, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QDrag, QFont
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSplitter,
    QVBoxLayout, QWidget,
)

from . import engine as eng

AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aif", ".aiff", ".wma", ".alac"}

STEM_COLORS = {
    "vocals": "#e8557a", "drums": "#f0a030", "bass": "#4fa3e0", "guitar": "#7ac74f",
    "piano": "#b07ae0", "other": "#9aa3ad", "instrumental": "#46c2b5", "residual": "#666e78",
    "lead vocals": "#ff6f91", "backing vocals": "#c94f72",
    "kick": "#f5b041", "snare": "#e59866", "toms": "#d68910", "hi-hat": "#f7dc6f",
    "ride": "#eacd76", "crash": "#d4ac0d",
}


def stem_color(name: str) -> str:
    if name in STEM_COLORS:
        return STEM_COLORS[name]
    for k, c in STEM_COLORS.items():      # "lead vocals dry" -> "lead vocals"
        if name.startswith(k):
            return c
    return "#888"

STYLE = """
QWidget { background: #17191d; color: #e6e8eb; font-size: 10pt; }
QGroupBox { border: 1px solid #2c3036; border-radius: 6px; margin-top: 14px; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9aa3ad; }
QPushButton { background: #262a30; border: 1px solid #353a42; border-radius: 5px; padding: 6px 12px; }
QPushButton:hover { background: #30353d; }
QPushButton:disabled { color: #5c636b; }
QPushButton#primary { background: #3d6ff0; border: none; font-weight: 600; padding: 8px 18px; }
QPushButton#primary:hover { background: #5280ff; }
QPushButton#primary:disabled { background: #2a3550; color: #7d8697; }
QComboBox, QListWidget, QPlainTextEdit { background: #1e2125; border: 1px solid #2c3036; border-radius: 5px; padding: 4px; }
QListWidget::item:selected { background: #2b3d66; }
QProgressBar { background: #1e2125; border: 1px solid #2c3036; border-radius: 5px; text-align: center; height: 18px; }
QProgressBar::chunk { background: #3d6ff0; border-radius: 4px; }
QFrame#stemRow { background: #1e2125; border: 1px solid #2c3036; border-radius: 6px; }
QFrame#stemRow:hover { border-color: #3d6ff0; }
QLabel { background: transparent; }
QLabel#muted { color: #8a929c; }
QScrollArea > QWidget > QWidget { background: transparent; }
"""


# --------------------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------------------

class QtLogHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            self.signal.emit(self.format(record))
        except Exception:
            pass


class Worker(QObject):
    progress = Signal(int, float, str)        # job index, fraction, message
    job_done = Signal(int, object)            # job index, Result
    job_failed = Signal(int, str)
    all_done = Signal()
    log = Signal(str)

    def __init__(self, engine: eng.Engine, jobs: list[eng.Job]):
        super().__init__()
        self.engine = engine
        self.jobs = jobs
        self._stop = False

    def stop(self):
        self._stop = True
        self.engine.cancel()

    def run(self):
        handler = QtLogHandler(self.log)
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            for i, job in enumerate(self.jobs):
                if self._stop:
                    break
                try:
                    res = self.engine.run(job, lambda f, m, i=i: self.progress.emit(i, f, m))
                    self.job_done.emit(i, res)
                except eng.Cancelled:
                    self.job_failed.emit(i, "Cancelled")
                    break
                except Exception as e:
                    self.log.emit(traceback.format_exc())
                    self.job_failed.emit(i, str(e))
        finally:
            root.removeHandler(handler)
            self.all_done.emit()


# --------------------------------------------------------------------------------------
# Stem row (play / save / drag-to-DAW)
# --------------------------------------------------------------------------------------

class StemRow(QFrame):
    play_requested = Signal(object)   # self

    def __init__(self, name: str, path: Path, full: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("stemRow")
        self.name, self.path = name, path
        self._drag_start = None
        self.setToolTip("Drag this row straight into your DAW or a folder")

        info = sf.info(str(path))
        try:
            data, _ = sf.read(str(path), dtype="float32", always_2d=True)
            peak = float(np.max(np.abs(data))) if data.size else 0.0
            rms = float(np.sqrt(np.mean(data**2))) if data.size else 0.0
        except Exception:
            peak = rms = 0.0
        peak_db = eng.db(peak)
        rms_db = eng.db(rms)

        swatch = QLabel()
        swatch.setFixedSize(6, 34)
        swatch.setStyleSheet(f"background:{stem_color(name)}; border-radius:3px;")

        title = QLabel(name[:1].upper() + name[1:])
        if full:
            title.setText(title.text() + "   <span style='color:#8a929c;font-weight:normal;font-size:9pt'>"
                          "full stem - also split into the stems below</span>")
        f = QFont(); f.setPointSize(11); f.setBold(True)
        title.setFont(f)
        meta = QLabel(f"{info.samplerate/1000:g} kHz · {info.subtype_info} · peak {peak_db:+.1f} dBFS · RMS {rms_db:.1f} dBFS")
        meta.setObjectName("muted")
        if rms_db < -60 and not full:
            title.setText(title.text() + "   <span style='color:#8a929c;font-weight:normal;font-size:9pt'>near-silent - probably not in this song</span>")

        text = QVBoxLayout(); text.setSpacing(0)
        text.addWidget(title); text.addWidget(meta)

        self.play_btn = QPushButton("▶  Play")
        self.play_btn.setFixedWidth(90)
        self.play_btn.clicked.connect(lambda: self.play_requested.emit(self))
        save_btn = QPushButton("Save as…")
        save_btn.clicked.connect(self.save_as)
        show_btn = QPushButton("Show")
        show_btn.clicked.connect(lambda: reveal_in_explorer(self.path))

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.addWidget(swatch)
        lay.addLayout(text, 1)
        lay.addWidget(self.play_btn)
        lay.addWidget(save_btn)
        lay.addWidget(show_btn)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMinimumHeight(max(self.sizeHint().height(), 56))

    def set_playing(self, playing: bool):
        self.play_btn.setText("■  Stop" if playing else "▶  Play")

    def save_as(self):
        start = str(Path(QSettings().value("last_save_dir", str(Path.home()))) / self.path.name)
        filters = "Same as source (*%s);;WAV 32-bit float (*.wav);;WAV 24-bit (*.wav);;FLAC 24-bit (*.flac)" % self.path.suffix
        dest, chosen = QFileDialog.getSaveFileName(self, f"Save {self.name}", start, filters)
        if not dest:
            return
        dest = Path(dest)
        QSettings().setValue("last_save_dir", str(dest.parent))
        try:
            if chosen.startswith("Same"):
                shutil.copy2(self.path, dest)
            else:
                fmt = chosen.split(" (")[0]
                data, sr = sf.read(str(self.path), dtype="float32", always_2d=True)
                eng.write_audio(dest, data.T, sr, fmt)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    # drag the file out into a DAW / Explorer
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_start = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag_start is None or not (e.buttons() & Qt.LeftButton):
            return
        if (e.position().toPoint() - self._drag_start).manhattanLength() < QApplication.startDragDistance():
            return
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(self.path))])
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)
        self._drag_start = None


def reveal_in_explorer(path: Path):
    if os.name == "nt":
        import subprocess
        subprocess.Popen(["explorer", "/select,", str(path)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))


# --------------------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------------------

@dataclass
class SongEntry:
    path: Path
    status: str = "Queued"
    result: eng.Result | None = None


class DropList(QListWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DropOnly)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = []
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.is_dir():
                paths += [q for q in sorted(p.iterdir()) if q.suffix.lower() in AUDIO_EXT]
            elif p.suffix.lower() in AUDIO_EXT:
                paths.append(p)
        self.files_dropped.emit(paths)
        e.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from . import __version__
        self.setWindowTitle(f"Stem Splitter {__version__}")
        self.resize(1100, 720)
        self.settings = QSettings()
        self.engine = eng.Engine()
        self.songs: list[SongEntry] = []
        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self.running_indices: list[int] = []

        self.player: QMediaPlayer | None = None     # created on first Play
        self.audio_out = None
        self.playing_row: StemRow | None = None

        self._build_ui()
        self.device_label.setText("Device: " + eng.Engine.device_info())

    # -- UI -----------------------------------------------------------------------
    def _build_ui(self):
        # Left: queue + settings
        self.queue = DropList()
        self.queue.files_dropped.connect(self.add_files)
        self.queue.currentRowChanged.connect(self.show_song)
        self.queue.setMinimumWidth(300)

        add_btn = QPushButton("Add songs…"); add_btn.clicked.connect(self.pick_files)
        rm_btn = QPushButton("Remove"); rm_btn.clicked.connect(self.remove_selected)
        qbtns = QHBoxLayout(); qbtns.addWidget(add_btn); qbtns.addWidget(rm_btn)

        hint = QLabel("Drop audio files or folders here")
        hint.setObjectName("muted")

        self.preset_cb = QComboBox()
        for p in eng.PRESETS.values():
            self.preset_cb.addItem(p.label, p.key)
            self.preset_cb.setItemData(self.preset_cb.count() - 1, p.description, Qt.ToolTipRole)
        self.preset_desc = QLabel(); self.preset_desc.setWordWrap(True); self.preset_desc.setObjectName("muted")

        self.vocal_cb = QComboBox()
        for k, v in eng.VOCAL_MODELS.items():
            self.vocal_cb.addItem(v, k)

        self.quality_cb = QComboBox(); self.quality_cb.addItems(list(eng.QUALITY))
        self.quality_cb.setToolTip("More overlap / shifts = fewer artefacts, longer processing")
        self.format_cb = QComboBox(); self.format_cb.addItems(list(eng.OUTPUT_FORMATS))

        self.residual_cb = QComboBox()
        self.residual_cb.addItem("Discard", "none")
        self.residual_cb.addItem("Save as extra 'residual' stem", "file")
        self.residual_cb.addItem("Add to 'other' (stems null exactly)", "other")
        self.residual_cb.setToolTip("Leftover = original mix minus the sum of all stems")

        self.out_dir_label = QLabel(); self.out_dir_label.setObjectName("muted"); self.out_dir_label.setWordWrap(True)
        out_btn = QPushButton("Change…"); out_btn.clicked.connect(self.pick_out_dir)
        out_row = QHBoxLayout(); out_row.addWidget(self.out_dir_label, 1); out_row.addWidget(out_btn)

        form = QFormLayout()
        form.addRow("Mode", self.preset_cb)
        form.addRow("", self.preset_desc)
        form.addRow("Vocal model", self.vocal_cb)
        form.addRow("Quality", self.quality_cb)
        form.addRow("Format", self.format_cb)
        form.addRow("Leftover", self.residual_cb)
        form.addRow("Output", out_row)
        settings_box = QGroupBox("Settings"); settings_box.setLayout(form)

        self.split_vocals_chk = QCheckBox("Lead / backing vocals")
        self.split_vocals_chk.setToolTip("Splits the vocal stem into the main vocal and harmonies / doubles / ad-libs")
        self.dereverb_chk = QCheckBox("Dry vocal + reverb (de-reverb)")
        self.dereverb_chk.setToolTip("Splits the (lead) vocal into a dry vocal and its reverb / echo tail")
        self.split_drums_chk = QCheckBox("Drum kit: kick, snare, toms, hi-hat, ride, crash")
        self.split_drums_chk.setToolTip("Splits the drum stem into kit pieces. Kick and snare are cleanest; "
                                        "cymbals bleed into each other more")
        self.keep_full_chk = QCheckBox("Also keep the full stems that were split")
        self.keep_full_chk.setChecked(True)
        extra = QVBoxLayout()
        for c in (self.split_vocals_chk, self.dereverb_chk, self.split_drums_chk, self.keep_full_chk):
            extra.addWidget(c)
        extra_hint = QLabel("Each extra split adds one model pass.")
        extra_hint.setObjectName("muted")
        extra.addWidget(extra_hint)
        extra_box = QGroupBox("Extra splits"); extra_box.setLayout(extra)

        self.go_btn = QPushButton("Split stems"); self.go_btn.setObjectName("primary")
        self.go_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.setEnabled(False)
        go_row = QHBoxLayout(); go_row.addWidget(self.go_btn, 1); go_row.addWidget(self.cancel_btn)

        self.progress = QProgressBar(); self.progress.setRange(0, 1000)
        self.status = QLabel("Ready"); self.status.setObjectName("muted")
        self.device_label = QLabel(); self.device_label.setObjectName("muted"); self.device_label.setWordWrap(True)

        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Songs</b>"))
        left.addWidget(hint)
        left.addWidget(self.queue, 1)
        left.addLayout(qbtns)
        left.addWidget(settings_box)
        left.addWidget(extra_box)
        left.addLayout(go_row)
        left.addWidget(self.progress)
        left.addWidget(self.status)
        left.addWidget(self.device_label)
        left_w = QWidget(); left_w.setLayout(left)

        # Right: stems of selected song + log
        self.song_title = QLabel("<b>Stems</b>")
        self.stems_box = QVBoxLayout(); self.stems_box.addStretch(1)
        stems_w = QWidget(); stems_w.setLayout(self.stems_box)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(stems_w)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.open_folder_btn = QPushButton("Open output folder")
        self.open_folder_btn.clicked.connect(self.open_song_folder)
        self.open_folder_btn.setEnabled(False)
        self.export_all_btn = QPushButton("Export all to…")
        self.export_all_btn.clicked.connect(self.export_all)
        self.export_all_btn.setEnabled(False)
        top = QHBoxLayout(); top.addWidget(self.song_title, 1)
        top.addWidget(self.export_all_btn); top.addWidget(self.open_folder_btn)

        self.log_view = QPlainTextEdit(); self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        f = QFont("Consolas"); f.setPointSize(9); self.log_view.setFont(f)

        right_split = QSplitter(Qt.Vertical)
        stems_panel = QWidget(); sp = QVBoxLayout(stems_panel); sp.setContentsMargins(0, 0, 0, 0)
        sp.addLayout(top); sp.addWidget(scroll, 1)
        right_split.addWidget(stems_panel)
        right_split.addWidget(self.log_view)
        right_split.setSizes([480, 180])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left_w); split.addWidget(right_split)
        split.setSizes([360, 740])
        self.setCentralWidget(split)

        # restore settings
        self._restore_combo(self.preset_cb, "preset", data=True)
        self._restore_combo(self.vocal_cb, "vocal_model", data=True)
        self._restore_combo(self.quality_cb, "quality")
        self._restore_combo(self.format_cb, "format")
        self._restore_combo(self.residual_cb, "residual", data=True)
        for key, chk in self._checks().items():
            v = self.settings.value(key)
            if v is not None:
                chk.setChecked(str(v).lower() in ("true", "1"))
        default_out = Path.home() / "Music" / "Stem Splitter"
        self.out_dir = Path(self.settings.value("out_dir", str(default_out)))
        self.out_dir_label.setText(str(self.out_dir))
        self.preset_cb.currentIndexChanged.connect(self._preset_changed)
        self._preset_changed()

    def _restore_combo(self, cb: QComboBox, key: str, data=False):
        v = self.settings.value(key)
        if v is None:
            return
        i = cb.findData(v) if data else cb.findText(v)
        if i >= 0:
            cb.setCurrentIndex(i)

    def _save_settings(self):
        s = self.settings
        s.setValue("preset", self.preset_cb.currentData())
        s.setValue("vocal_model", self.vocal_cb.currentData())
        s.setValue("quality", self.quality_cb.currentText())
        s.setValue("format", self.format_cb.currentText())
        s.setValue("residual", self.residual_cb.currentData())
        s.setValue("out_dir", str(self.out_dir))
        for key, chk in self._checks().items():
            s.setValue(key, chk.isChecked())

    def _checks(self) -> dict:
        return {"split_vocals": self.split_vocals_chk, "dereverb": self.dereverb_chk,
                "split_drums": self.split_drums_chk, "keep_full": self.keep_full_chk}

    def _preset_changed(self):
        p = eng.PRESETS[self.preset_cb.currentData()]
        self.preset_desc.setText(p.description)
        self.vocal_cb.setEnabled(p.key in ("max6", "vox2"))
        self.split_drums_chk.setEnabled("drums" in p.stems)

    # -- queue ----------------------------------------------------------------------
    def pick_files(self):
        start = self.settings.value("last_open_dir", str(Path.home()))
        exts = " ".join("*" + e for e in sorted(AUDIO_EXT))
        files, _ = QFileDialog.getOpenFileNames(self, "Add songs", start, f"Audio ({exts});;All files (*)")
        if files:
            self.settings.setValue("last_open_dir", str(Path(files[0]).parent))
            self.add_files([Path(f) for f in files])

    def add_files(self, paths: list[Path]):
        known = {s.path for s in self.songs}
        for p in paths:
            if p in known:
                continue
            self.songs.append(SongEntry(p))
            self.queue.addItem(QListWidgetItem(self._item_text(self.songs[-1])))
        if self.queue.currentRow() < 0 and self.songs:
            self.queue.setCurrentRow(0)

    def remove_selected(self):
        r = self.queue.currentRow()
        if r < 0 or r in self.running_indices:
            return
        self.queue.takeItem(r)
        del self.songs[r]

    @staticmethod
    def _item_text(s: SongEntry) -> str:
        return f"{s.path.name}\n   {s.status}"

    def _refresh_item(self, i: int):
        self.queue.item(i).setText(self._item_text(self.songs[i]))

    def pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", str(self.out_dir))
        if d:
            self.out_dir = Path(d)
            self.out_dir_label.setText(d)
            self._save_settings()

    # -- processing -----------------------------------------------------------------
    def start(self):
        todo = [i for i, s in enumerate(self.songs) if s.result is None]
        if not todo:
            sel = self.queue.currentRow()
            if sel < 0:
                QMessageBox.information(self, "Stem Splitter", "Add one or more songs first.")
                return
            todo = [sel]   # re-run the selected song with new settings
        self._save_settings()
        jobs = []
        for i in todo:
            s = self.songs[i]
            jobs.append(eng.Job(
                input_path=s.path,
                output_dir=self.out_dir / s.path.stem,
                preset=self.preset_cb.currentData(),
                vocal_model=self.vocal_cb.currentData(),
                quality=self.quality_cb.currentText(),
                output_format=self.format_cb.currentText(),
                residual=self.residual_cb.currentData(),
                split_vocals=self.split_vocals_chk.isChecked(),
                dereverb=self.dereverb_chk.isChecked(),
                split_drums=self.split_drums_chk.isChecked() and self.split_drums_chk.isEnabled(),
                keep_full=self.keep_full_chk.isChecked(),
            ))
            s.status = "Waiting…"; self._refresh_item(i)
        self.running_indices = todo

        self.thread = QThread(self)
        self.worker = Worker(self.engine, jobs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.job_done.connect(self._on_done)
        self.worker.job_failed.connect(self._on_failed)
        self.worker.log.connect(self.log_view.appendPlainText)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.all_done.connect(self.thread.quit)
        self.go_btn.setEnabled(False); self.cancel_btn.setEnabled(True)
        self.thread.start()

    def cancel(self):
        if self.worker:
            self.worker.stop()
            self.status.setText("Cancelling after the current step…")

    def _on_progress(self, j: int, frac: float, msg: str):
        i = self.running_indices[j]
        n = len(self.running_indices)
        self.progress.setValue(int(1000 * (j + frac) / n))
        self.status.setText(f"[{j+1}/{n}] {self.songs[i].path.name}: {msg}")
        self.songs[i].status = msg; self._refresh_item(i)

    def _on_done(self, j: int, res: eng.Result):
        i = self.running_indices[j]
        s = self.songs[i]
        s.result = res
        n = len(res.stems) - len(res.full_stems)
        s.status = f"✓ {n} stems · leftover {res.residual_db:.0f} dB"
        self._refresh_item(i)
        if self.queue.currentRow() == i:
            self.show_song(i)

    def _on_failed(self, j: int, err: str):
        i = self.running_indices[j]
        self.songs[i].status = f"✗ {err}"
        self._refresh_item(i)
        if err != "Cancelled":
            self.log_view.appendPlainText(f"ERROR ({self.songs[i].path.name}): {err}")

    def _on_all_done(self):
        self.go_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        self.status.setText("Finished")
        for i in self.running_indices:
            if self.songs[i].result is None and not self.songs[i].status.startswith("✗"):
                self.songs[i].status = "Queued"; self._refresh_item(i)
        self.running_indices = []

    # -- results --------------------------------------------------------------------
    def _clear_stems(self):
        if self.player:
            self.player.stop()
        self.playing_row = None
        while self.stems_box.count() > 1:
            w = self.stems_box.takeAt(0).widget()
            if w:
                w.deleteLater()

    def show_song(self, i: int):
        self._clear_stems()
        if i < 0 or i >= len(self.songs):
            self.song_title.setText("<b>Stems</b>")
            return
        s = self.songs[i]
        self.song_title.setText(f"<b>{s.path.name}</b>")
        has = s.result is not None
        self.open_folder_btn.setEnabled(has)
        self.export_all_btn.setEnabled(has)
        if not has:
            lbl = QLabel("Not split yet. Press <b>Split stems</b>.")
            lbl.setObjectName("muted")
            self.stems_box.insertWidget(0, lbl)
            return
        for k, (name, path) in enumerate(s.result.stems.items()):
            row = StemRow(name, path, full=name in s.result.full_stems)
            row.play_requested.connect(self.toggle_play)
            self.stems_box.insertWidget(k, row)

    def _ensure_player(self):
        if self.player is None:
            self.player = QMediaPlayer(self)
            self.audio_out = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_out)
            self.player.playbackStateChanged.connect(self._on_playback_state)

    def toggle_play(self, row: StemRow):
        self._ensure_player()
        if self.playing_row is row and self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.stop()
            return
        if self.playing_row:
            self.playing_row.set_playing(False)
        self.playing_row = row
        self.player.setSource(QUrl.fromLocalFile(str(row.path)))
        self.player.play()
        row.set_playing(True)

    def _on_playback_state(self, state):
        if state != QMediaPlayer.PlayingState and self.playing_row:
            self.playing_row.set_playing(False)

    def _current_result(self):
        i = self.queue.currentRow()
        return self.songs[i] if 0 <= i < len(self.songs) and self.songs[i].result else None

    def open_song_folder(self):
        s = self._current_result()
        if s:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(next(iter(s.result.stems.values())).parent)))

    def export_all(self):
        s = self._current_result()
        if not s:
            return
        d = QFileDialog.getExistingDirectory(self, "Copy all stems to", str(self.out_dir))
        if not d:
            return
        for p in s.result.stems.values():
            shutil.copy2(p, Path(d) / p.name)
        self.status.setText(f"Copied {len(s.result.stems)} stems to {d}")

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(3000)
        self._save_settings()
        super().closeEvent(e)


def _ensure_std_streams():
    """Under pythonw.exe (no console) sys.stdout/stderr are None, and libraries that print
    progress bars (tqdm during model downloads) crash. Send that output to a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_path = eng.app_data_dir() / "app.log"
    f = open(log_path, "w", encoding="utf-8", errors="replace", buffering=1)
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f


def main():
    _ensure_std_streams()
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("StemSplitter.App")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = QApplication(sys.argv)
    app.setOrganizationName("StemSplitter")
    app.setApplicationName("Stem Splitter")
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
