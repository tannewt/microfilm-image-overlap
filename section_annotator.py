#!/usr/bin/env python3
"""Review plan-v2 pages: mark contributing sections and confirm gaps.

Plan v2 writes one TOML per page with a list of contributing scan sections
and any uncovered film rows (``gaps``).  This viewer composes that page with
``compose_pages.py``'s existing section assembler, draws the section and gap
bands over it, and records review decisions in a separate
``section_overrides.json`` beside the plan tree:

    {
      "pages": {
        "page0062-56s3-57s1-gap": {
          "status": "gap_accepted",          # or unreviewed/ok/needs_fix/skip
          "sections": {
            "..._0056.tif|3": {"keep": true},
            "..._0057.tif|1": {"keep": true}
          },
          "gaps": {
            "1234.0-1300.0": {"status": "confirmed"}  # or unreviewed/not_gap
          },
          "note": ""
        }
      },
      "links": {
        "..._0056.tif|3": ["..._0057.tif|1"]
      },
      "duplicates": {
        "P72-.../08490_.../57": {"of": "P72-.../08490_.../56",
                                 "note": "duplicate exposure"}
      }
    }

Review decisions live in the override file, so they survive a re-plan.  The
replan action rewrites the kept page's generated TOML and absorbs stale
short/gap duplicates instead of leaving them behind.

The **Sections & bars** tab explodes the page into its contributing scan
sections.  Select one to see the full scan with its manual/auto bars, all
sections, and the exact page band highlighted.  Bars can be added, adjusted,
or accepted there; that writes the shared bar annotation JSON and marks the
page `needs_fix`; **Replan** (R) then rebuilds just this page with plan_v2 and
loads the new version.  The composed page lives in the **Page** tab.

The **Scan map** tab shows two consecutive scans at a time (navigate the
neighbouring pairs with the buttons).  Sections whose film-row intervals
overlap are linked as a hint; drag from a section in one scan to a section in
the other to mark them as the same.  Drag the same pair again to unlink, or
use **Clear pair links**.  Manual links are magenta and stored in
``section_overrides.json`` under ``links``.

Usage:
    python section_annotator.py plans/v2/P72-2707-CENLIB9102
    python section_annotator.py plans/v2/P72-2707-CENLIB9102 \
        --overrides plans/v2/P72-2707-CENLIB9102/section_overrides.json

Controls
--------
    Ctrl+O        open a plan-v2 directory
    Ctrl+S        save overrides (autosaves too)
    Right / Space next page
    Left          previous page
    N             next page needing review
    R             replan the current page with plan_v2 and reload it
    Ctrl+0        fit page
    Ctrl+1        show the composed page tab
    Ctrl+2        show the sections & bars tab
    Ctrl+3        show the scan/section map tab
    Ctrl+wheel    zoom
    drag          pan
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import compose_pages as cp
import plan_v2 as pv2
import bar_annotator as ba
from pair_annotator import (
    AnnotationStore,
    OcrLibrary,
    default_annotation_path,
    scan_index,
)

REPO = Path(__file__).resolve().parent
DEFAULT_PLAN_ROOT = REPO / "plans" / "v2"

STATUS_UNREVIEWED = "unreviewed"
STATUS_OK = "ok"
STATUS_NEEDS_FIX = "needs_fix"
STATUS_GAP_ACCEPTED = "gap_accepted"
STATUS_SKIP = "skip"
PAGE_STATUSES = {
    STATUS_UNREVIEWED: "Unreviewed",
    STATUS_OK: "OK / accepted",
    STATUS_NEEDS_FIX: "Needs fix",
    STATUS_GAP_ACCEPTED: "Gap accepted",
    STATUS_SKIP: "Skip page",
}
GAP_UNREVIEWED = "unreviewed"
GAP_CONFIRMED = "confirmed"
GAP_NOT_GAP = "not_gap"
GAP_STATUSES = {
    GAP_UNREVIEWED: "Unreviewed",
    GAP_CONFIRMED: "Confirmed real gap",
    GAP_NOT_GAP: "Not a gap",
}


# --------------------------------------------------------------------------- #
# plans
# --------------------------------------------------------------------------- #

@dataclass
class PagePlan:
    path: Path
    label: str
    kind: str
    page_index: int
    source_dir: Path
    film: tuple[float, float]
    sections: list[dict]
    gaps: list[list[float]]
    review: dict
    inventory: dict
    reverse: bool = False

    def section_key(self, sec: dict) -> str:
        return f"{sec.get('scan')}|{sec.get('section')}"

    @staticmethod
    def gap_key(gap) -> str:
        lo, hi = float(gap[0]), float(gap[1])
        return f"{lo:.1f}-{hi:.1f}"


def load_plan(path: Path) -> PagePlan | None:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    page = data.get("page", {}) or {}
    auto = data.get("auto", {}) or {}
    if "film" not in auto:
        return None
    src = page.get("source_dir")
    if not src:
        return None
    return PagePlan(
        path=path,
        label=page.get("label", path.stem),
        kind=page.get("kind", "page"),
        page_index=int(page.get("page_index", 0)),
        source_dir=Path(src),
        film=(float(auto["film"][0]), float(auto["film"][1])),
        sections=list(auto.get("sections") or []),
        gaps=[list(map(float, g)) for g in (auto.get("gaps") or [])],
        review=dict(data.get("review", {}) or {}),
        inventory=dict(data.get("inventory", {}) or {}),
        reverse=bool(page.get("reverse", False)),
    )


def load_plans(plan_dir: Path) -> list[PagePlan]:
    files = list(plan_dir.rglob("*.toml"))
    plans = [p for p in (load_plan(f) for f in files) if p is not None]
    plans.sort(key=lambda p: (p.page_index, p.label))
    return plans


# --------------------------------------------------------------------------- #
# overrides
# --------------------------------------------------------------------------- #

class OverrideStore:
    VERSION = 1

    def __init__(self, path: Path):
        self.path = Path(path)
        self.pages: dict[str, dict] = {}
        # Manual "same section" links across scans, keyed by
        # "<scan filename>|<section number>" and stored symmetrically.
        self.links: dict[str, list[str]] = {}
        # Scans explicitly excluded from planning as duplicate exposures,
        # keyed by "<roll>/<prefix>/<index>".
        self.duplicates: dict[str, dict] = {}

    def load(self) -> None:
        if not self.path.is_file():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.pages = dict(data.get("pages") or {})
        self.links = {str(k): [str(v) for v in vals]
                      for k, vals in (data.get("links") or {}).items()}
        self.duplicates = {str(k): dict(v)
                           for k, v in (data.get("duplicates") or {}).items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        kept = {}
        for label, rec in self.pages.items():
            if (rec.get("status", STATUS_UNREVIEWED) != STATUS_UNREVIEWED
                    or rec.get("sections") or rec.get("gaps")
                    or rec.get("note")):
                kept[label] = rec
        self.pages = kept
        links = {}
        for key, partners in self.links.items():
            clean = sorted({p for p in partners if p and p != key})
            if clean:
                links[key] = clean
        self.links = links
        duplicates = {}
        for key, rec in self.duplicates.items():
            if key:
                duplicates[key] = dict(rec)
        self.duplicates = duplicates
        data = {"version": self.VERSION,
                "pages": {k: self.pages[k] for k in sorted(self.pages)}}
        if links:
            data["links"] = links
        if duplicates:
            data["duplicates"] = {k: duplicates[k] for k in sorted(duplicates)}
        tmp = self.path.with_suffix(".part.json")
        tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def is_linked(self, a: str, b: str) -> bool:
        return b in (self.links.get(a) or [])

    def toggle_link(self, a: str, b: str) -> bool:
        """Link or unlink two sections; return True when a link was added."""
        if a == b:
            return False
        if self.is_linked(a, b):
            self.links[a] = [x for x in self.links.get(a) or [] if x != b]
            self.links[b] = [x for x in self.links.get(b) or [] if x != a]
            if not self.links[a]:
                self.links.pop(a, None)
            if not self.links[b]:
                self.links.pop(b, None)
            return False
        self.links.setdefault(a, []).append(b)
        self.links.setdefault(b, []).append(a)
        return True

    def clear_links_between(self, left: set[str], right: set[str]) -> bool:
        changed = False
        for a in left:
            for b in right:
                if self.is_linked(a, b):
                    self.toggle_link(a, b)
                    changed = True
        return changed

    @staticmethod
    def _empty_page() -> dict:
        return {"status": STATUS_UNREVIEWED, "sections": {}, "gaps": {},
                "note": "", "updated": ""}

    def get_page(self, label: str) -> dict:
        return self.pages.get(label) or self._empty_page()

    def page(self, label: str) -> dict:
        return self.pages.setdefault(label, self._empty_page())

    def get_section(self, label: str, key: str) -> dict:
        rec = self.get_page(label).get("sections") or {}
        return rec.get(key) or {"keep": True, "note": ""}

    def section(self, label: str, key: str) -> dict:
        rec = self.page(label)
        return rec.setdefault("sections", {}).setdefault(
            key, {"keep": True, "note": ""})

    def get_gap(self, label: str, key: str) -> dict:
        rec = self.get_page(label).get("gaps") or {}
        return rec.get(key) or {"status": GAP_UNREVIEWED, "note": ""}

    def gap(self, label: str, key: str) -> dict:
        rec = self.page(label)
        return rec.setdefault("gaps", {}).setdefault(
            key, {"status": GAP_UNREVIEWED, "note": ""})

    def mark_duplicate(self, scan: str, of: str = "",
                       note: str = "") -> None:
        rec = self.duplicates.setdefault(scan, {})
        if of:
            rec["of"] = of
        if note:
            rec["note"] = note
        rec.setdefault("note", "")
        rec["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    def is_duplicate(self, scan: str) -> bool:
        return scan in self.duplicates

    def duplicate_of(self, scan: str) -> str:
        return str((self.duplicates.get(scan) or {}).get("of") or "")

    def unmark_duplicate(self, scan: str) -> bool:
        return self.duplicates.pop(scan, None) is not None

    def touch(self, label: str) -> None:
        self.page(label)["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- #
# preview view
# --------------------------------------------------------------------------- #

class PageView(QGraphicsView):
    """Composite image with section/gap bands drawn over it."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing
                            | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#181818")))
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._overlays: list[QGraphicsItem] = []
        self._image_size = QSize()

    def clear(self):
        self._scene.clear()
        self._overlays = []
        self._pixmap_item = None

    def set_image(self, image: np.ndarray, page: PagePlan,
                  sections: list[dict], section_state: dict[str, bool],
                  gaps: list[list[float]], gap_state: dict[str, str]):
        self._scene.clear()
        self._overlays = []
        self._pixmap_item = None
        if image is None:
            image = np.zeros((max(200, int(page.film[1] - page.film[0])), 1000),
                             dtype=np.uint8)
        if image.ndim == 2:
            h, w = image.shape
            qimg = QImage(image.data, w, h, image.strides[0],
                          QImage.Format_Grayscale8).copy()
        else:
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, rgb.strides[0],
                          QImage.Format_RGB888).copy()
        self._image_size = QSize(w, h)
        item = self._scene.addPixmap(QPixmap.fromImage(qimg))
        item.setPos(0, 0)
        self._pixmap_item = item
        self._scene.setSceneRect(QRectF(0, 0, w, h))

        colors = ["#5aa9ff", "#33d17a", "#ffb340", "#ff7ab6", "#b38cff",
                  "#6ee7e7"]
        base = page.film[0]
        for i, sec in enumerate(sections):
            key = page.section_key(sec)
            keep = section_state.get(key, True)
            y0 = max(0.0, sec["film"][0] - base)
            y1 = min(float(h), sec["film"][1] - base)
            if y1 <= y0:
                continue
            color = QColor(colors[i % len(colors)])
            color.setAlpha(38 if keep else 20)
            rect = QGraphicsRectItem(0, y0, w, y1 - y0)
            rect.setBrush(QBrush(color))
            rect.setPen(QPen(QColor(colors[i % len(colors)]), 2,
                             Qt.SolidLine if keep else Qt.DashLine))
            rect.setZValue(10)
            self._scene.addItem(rect)
            self._overlays.append(rect)
            label = f"{sec['index']}s{sec['section']}" + ("" if keep else " SKIP")
            text = QGraphicsSimpleTextItem(label)
            font = QFont()
            font.setBold(True)
            font.setPointSize(9)
            text.setFont(font)
            text.setBrush(QBrush(color.lighter(140)))
            text.setPos(10, y0 + 4)
            text.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            text.setZValue(20)
            self._scene.addItem(text)
            self._overlays.append(text)

        for gap in gaps:
            key = PagePlan.gap_key(gap)
            state = gap_state.get(key, GAP_UNREVIEWED)
            y0 = max(0.0, gap[0] - base)
            y1 = min(float(h), gap[1] - base)
            if y1 <= y0:
                continue
            if state == GAP_CONFIRMED:
                color, text_color = QColor(255, 40, 40, 90), "#ff8080"
            elif state == GAP_NOT_GAP:
                color, text_color = QColor(90, 90, 90, 60), "#bbbbbb"
            else:
                color, text_color = QColor(255, 140, 40, 55), "#ffb340"
            rect = QGraphicsRectItem(0, y0, w, y1 - y0)
            rect.setBrush(QBrush(color))
            rect.setPen(QPen(QColor(text_color), 2, Qt.DashLine))
            rect.setZValue(15)
            self._scene.addItem(rect)
            self._overlays.append(rect)
            text = QGraphicsSimpleTextItem(f"GAP {state}")
            font = QFont()
            font.setBold(True)
            font.setPointSize(9)
            text.setFont(font)
            text.setBrush(QBrush(QColor(text_color)))
            text.setPos(10, y0 + 4)
            text.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            text.setZValue(20)
            self._scene.addItem(text)
            self._overlays.append(text)

        self.fit()

    def fit(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self.fit()

    def wheelEvent(self, event):  # noqa: N802
        delta = event.angleDelta().y()
        if delta:
            factor = 1.25 if delta > 0 else 0.8
            self.scale(factor, factor)
        event.accept()


# --------------------------------------------------------------------------- #
# section map
# --------------------------------------------------------------------------- #

class SectionMapView(QGraphicsView):
    """Two consecutive scans side by side with their sections linked.

    Sections whose film-coordinate intervals overlap are connected as a
    hint.  Drag from a section in one scan to a section in the other to
    record a manual "same section" link; drag the same pair again to remove
    it.  Manual links are magenta and persist in ``section_overrides.json``.
    """

    linkDropped = Signal(str, str)      # section keys: left, right

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing
                            | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#181818")))
        self.setMouseTracking(True)
        self._content = QRectF()
        self._section_hits: list[tuple[QRectF, dict]] = []
        self._drag_from: dict | None = None
        self._temp_line: QGraphicsLineItem | None = None

    def clear(self):
        self._end_drag()
        self._scene.clear()
        self._content = QRectF()
        self._section_hits = []

    def fit(self) -> None:
        if not self._content.isEmpty():
            self.fitInView(self._content, Qt.KeepAspectRatio)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self.fit()

    def wheelEvent(self, event):  # noqa: N802
        delta = event.angleDelta().y()
        if delta:
            factor = 1.25 if delta > 0 else 0.8
            self.scale(factor, factor)
        event.accept()

    # ---------- drawing -------------------------------------------------

    @staticmethod
    def _section_key(panel: dict, number: int) -> str:
        return f"{panel.get('scan')}|{number}"

    def _draw_line(self, x1: float, y1: float, x2: float, y2: float,
                   color: QColor, width: float, style=Qt.SolidLine):
        line = QGraphicsLineItem(x1, y1, x2, y2)
        line.setPen(QPen(color, width, style))
        line.setZValue(25)
        self._scene.addItem(line)

    def set_panels(self, film: tuple[float, float], panels: list[dict],
                   links: dict[str, list[str]] | None = None):
        self._end_drag()
        self._scene.clear()
        self._content = QRectF()
        self._section_hits = []
        if not panels:
            return
        link_pairs = set()
        for a, partners in (links or {}).items():
            for b in partners:
                if a != b:
                    link_pairs.add(tuple(sorted((a, b))))
        linked_keys = {k for pair in link_pairs for k in pair}

        height = 1500.0
        gap = 300.0
        loaded = []
        for panel in panels[:2]:
            image = QImage(str(panel["path"]))
            if image.isNull():
                continue
            scale = height / max(1, image.height())
            width = max(40.0, image.width() * scale)
            loaded.append((panel, image, scale, width))
        if not loaded:
            return

        x = 0.0
        for panel, image, scale, width in loaded:
            panel["x"] = x
            panel["w"] = width
            panel["scale"] = scale
            x += width + gap
        total_w = x - gap
        geo_by_key: dict[str, dict] = {}

        for panel_index, (panel, image, scale, width) in enumerate(loaded):
            px = panel["x"]
            pixmap = QPixmap.fromImage(image.scaled(
                max(1, int(width)), max(1, int(height)),
                Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            item = self._scene.addPixmap(pixmap)
            item.setPos(px, 0)
            item.setZValue(0)
            border = QGraphicsRectItem(px, 0, width, height)
            border.setBrush(QBrush(Qt.NoBrush))
            border.setPen(QPen(QColor("#3a3a3a"), 1))
            border.setZValue(1)
            self._scene.addItem(border)

            py0 = (film[0] - panel["offset"]) * scale
            py1 = (film[1] - panel["offset"]) * scale
            py0 = max(0.0, min(height, py0))
            py1 = max(0.0, min(height, py1))
            if py1 > py0:
                band = QGraphicsRectItem(px, py0, width, py1 - py0)
                band.setBrush(QBrush(QColor(255, 255, 255, 10)))
                band.setPen(QPen(QColor(255, 255, 255, 70), 1, Qt.DashLine))
                band.setZValue(2)
                self._scene.addItem(band)

            used = panel.get("used") or {}
            for number, (lo, hi) in enumerate(panel.get("sections") or [], 1):
                y0 = max(0.0, min(height, lo * scale))
                y1 = max(0.0, min(height, hi * scale))
                if y1 <= y0:
                    continue
                key = self._section_key(panel, number)
                is_used = number in used
                keep = used.get(number, True)
                linked = key in linked_keys
                if linked:
                    fill, edge = QColor(214, 92, 255, 38), "#d65cff"
                    style = Qt.SolidLine
                elif is_used and keep:
                    fill, edge = QColor(51, 209, 122, 50), "#33d17a"
                    style = Qt.SolidLine
                elif is_used:
                    fill, edge = QColor(255, 138, 128, 45), "#ff8a80"
                    style = Qt.DashLine
                else:
                    fill, edge = QColor(90, 169, 255, 16), "#5aa9ff"
                    style = Qt.DotLine
                rect = QGraphicsRectItem(px, y0, width, y1 - y0)
                rect.setBrush(QBrush(fill))
                rect.setPen(QPen(QColor(edge), 2.0 if linked else 1.5, style))
                rect.setZValue(5)
                self._scene.addItem(rect)
                label = f"S{number}"
                if linked:
                    label += "  LINK"
                elif is_used:
                    label += "  PAGE"
                text = QGraphicsSimpleTextItem(label)
                font = QFont()
                font.setBold(True)
                font.setPointSize(9)
                text.setFont(font)
                text.setBrush(QBrush(QColor(edge).lighter(130)))
                text.setPos(px + 5, y0 + 3)
                text.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                text.setZValue(20)
                self._scene.addItem(text)
                geo = {
                    "key": key,
                    "panel": panel_index,
                    "number": number,
                    "rect": QRectF(px, y0, width, y1 - y0),
                    "x": px,
                    "w": width,
                    "y0": y0,
                    "y1": y1,
                    "cy": (y0 + y1) / 2.0,
                    "scale": scale,
                }
                geo_by_key[key] = geo
                self._section_hits.append((geo["rect"], geo))

            title = QGraphicsSimpleTextItem(
                f"scan {panel.get('index', '?')}  "
                f"{panel.get('scan', '')}")
            font = QFont()
            font.setBold(True)
            font.setPointSize(10)
            title.setFont(font)
            title.setBrush(QBrush(QColor("#e8e8e8")))
            title.setPos(px, -34)
            title.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            title.setZValue(30)
            self._scene.addItem(title)

        # Automatic overlap hints between the two panels.
        if len(loaded) == 2:
            lpanel, _, lscale, lwidth = loaded[0]
            rpanel, _, rscale, _ = loaded[1]
            for lnum, (llo, lhi) in enumerate(lpanel.get("sections") or [], 1):
                lkey = self._section_key(lpanel, lnum)
                lf0 = lpanel["offset"] + llo
                lf1 = lpanel["offset"] + lhi
                for rnum, (rlo, rhi) in enumerate(
                        rpanel.get("sections") or [], 1):
                    rkey = self._section_key(rpanel, rnum)
                    if tuple(sorted((lkey, rkey))) in link_pairs:
                        continue
                    rf0 = rpanel["offset"] + rlo
                    rf1 = rpanel["offset"] + rhi
                    if min(lf1, rf1) - max(lf0, rf0) <= 0:
                        continue
                    yl = ((llo + lhi) / 2.0) * lscale
                    yr = ((rlo + rhi) / 2.0) * rscale
                    lused = lnum in (lpanel.get("used") or {})
                    rused = rnum in (rpanel.get("used") or {})
                    if lused and rused:
                        color, alpha, width_ = QColor(51, 209, 122), 200, 2.5
                    elif lused or rused:
                        color, alpha, width_ = QColor(255, 179, 64), 170, 1.8
                    else:
                        color, alpha, width_ = QColor(90, 169, 255), 80, 1.0
                    color.setAlpha(alpha)
                    self._draw_line(lpanel["x"] + lwidth, yl,
                                    rpanel["x"], yr, color, width_)

            # Dashed red gap hint, unless the user has linked the pair.
            manual_pair = any(
                geo_by_key.get(k, {}).get("panel") == 0
                and geo_by_key.get(other, {}).get("panel") == 1
                for k in geo_by_key
                for other in (links or {}).get(k, []))
            if not manual_pair:
                mapped = any(
                    min(lpanel["offset"] + lhi, rpanel["offset"] + rhi)
                    - max(lpanel["offset"] + llo, rpanel["offset"] + rlo) > 0
                    for llo, lhi in lpanel.get("sections") or []
                    for rlo, rhi in rpanel.get("sections") or [])
                if not mapped:
                    lsecs = lpanel.get("sections") or []
                    rsecs = rpanel.get("sections") or []
                    lused = [(n, sec) for n, sec in enumerate(lsecs, 1)
                             if n in (lpanel.get("used") or {})]
                    rused = [(n, sec) for n, sec in enumerate(rsecs, 1)
                             if n in (rpanel.get("used") or {})]
                    best = None
                    for _lnum, (llo, lhi) in lused:
                        lf0 = lpanel["offset"] + llo
                        lf1 = lpanel["offset"] + lhi
                        for _rnum, (rlo, rhi) in rused:
                            rf0 = rpanel["offset"] + rlo
                            rf1 = rpanel["offset"] + rhi
                            distance = max(0.0, max(lf0, rf0) - min(lf1, rf1))
                            if best is None or distance < best[0]:
                                best = (distance, llo, lhi, rlo, rhi)
                    if best is not None:
                        _, llo, lhi, rlo, rhi = best
                        self._draw_line(
                            lpanel["x"] + lwidth,
                            ((llo + lhi) / 2.0) * lscale,
                            rpanel["x"],
                            ((rlo + rhi) / 2.0) * rscale,
                            QColor(255, 80, 80, 220), 2.0, Qt.DashLine)

        # Manual links on top, in magenta.
        for a, b in sorted(link_pairs):
            ga, gb = geo_by_key.get(a), geo_by_key.get(b)
            if ga is None or gb is None or ga["panel"] == gb["panel"]:
                continue
            left, right = (ga, gb) if ga["panel"] < gb["panel"] else (gb, ga)
            self._draw_line(left["x"] + left["w"], left["cy"],
                            right["x"], right["cy"],
                            QColor(214, 92, 255, 235), 4.0)

        self._content = QRectF(-20, -55, total_w + 40, height + 100)
        self._scene.setSceneRect(self._content)
        self.resetTransform()
        self.fit()

    # ---------- drag to link --------------------------------------------

    def _hit(self, scene_pos) -> dict | None:
        for rect, geo in reversed(self._section_hits):
            if rect.contains(scene_pos):
                return geo
        return None

    def _end_drag(self):
        if self._temp_line is not None:
            if self._temp_line.scene() is self._scene:
                self._scene.removeItem(self._temp_line)
            self._temp_line = None
        self._drag_from = None
        self.viewport().setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            geo = self._hit(self.mapToScene(event.position().toPoint()))
            if geo is not None:
                self._drag_from = geo
                x = geo["x"] + geo["w"] / 2.0
                y = geo["cy"]
                self._temp_line = QGraphicsLineItem(x, y, x, y)
                self._temp_line.setPen(QPen(QColor(214, 92, 255, 220), 2.0,
                                           Qt.DashLine))
                self._temp_line.setZValue(40)
                self._scene.addItem(self._temp_line)
                self.viewport().setCursor(Qt.CrossCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        pos = self.mapToScene(event.position().toPoint())
        if self._drag_from is not None and self._temp_line is not None:
            x = self._drag_from["x"] + self._drag_from["w"] / 2.0
            self._temp_line.setLine(x, self._drag_from["cy"], pos.x(), pos.y())
            event.accept()
            return
        if self._hit(pos) is not None:
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.viewport().setCursor(Qt.OpenHandCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if (event.button() == Qt.LeftButton
                and self._drag_from is not None):
            source = self._drag_from
            target = self._hit(self.mapToScene(event.position().toPoint()))
            self._end_drag()
            if (target is not None
                    and target["panel"] != source["panel"]):
                self.linkDropped.emit(source["key"], target["key"])
            event.accept()
            return
        super().mouseReleaseEvent(event)


# --------------------------------------------------------------------------- #
# window
# --------------------------------------------------------------------------- #

class MainWindow(QMainWindow):
    def __init__(self, plan_dir: Path | None = None,
                 overrides: Path | None = None,
                 annotations: Path | None = None,
                 plan_root: Path | None = None):
        super().__init__()
        self.setWindowTitle("Section / Gap Reviewer")
        self.resize(1500, 950)
        self.plan_dir: Path | None = None
        self.plans: list[PagePlan] = []
        self.visible: list[int] = []
        self.current: int | None = None
        self.overrides: OverrideStore | None = None
        self.plan_root = Path(plan_root).resolve() if plan_root else \
            REPO / "plans" / "ocr_1028"
        self.lru = cp._LRU(max_len=6)
        self._loading = False
        self._dirty = False
        self.annotations_path = Path(annotations) if annotations else None
        self.bar_store: AnnotationStore | None = None
        self.ocr_lib: OcrLibrary | None = None
        self.auto_cache: dict[str, tuple[list, tuple]] = {}
        self.component_page: PagePlan | None = None
        self.component_section: dict | None = None
        self.component_key: str | None = None
        self.component_path: Path | None = None
        self.component_manual: list = []
        self.component_auto: list = []
        self.component_bbox = (0, 0, 0, 0)
        self.component_selected: int | None = None
        self.component_selected_auto: int | None = None
        self.map_panels: list[dict] = []
        self.map_pair = 0
        self.map_film: tuple[float, float] = (0.0, 0.0)
        self.map_page_label: str | None = None

        self._autosave = QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(700)
        self._autosave.timeout.connect(self._autosave_now)

        self._make_actions()
        self._make_ui()
        if plan_dir is not None:
            self.load_plan_dir(Path(plan_dir), overrides)

    # ---------- UI -----------------------------------------------------

    def _act(self, text, shortcut, slot, checkable=False):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setCheckable(checkable)
        a.triggered.connect(slot)
        self.addAction(a)
        return a

    def _make_actions(self):
        self.act_open = self._act("&Open plan directory…", "Ctrl+O",
                                  self.open_dir)
        self.act_save = self._act("&Save overrides", "Ctrl+S", self.save)
        self.act_prev = self._act("Previous page", "Left", lambda: self.step(-1))
        self.act_next = self._act("Next page", "Space", lambda: self.step(1))
        self.act_next_review = self._act("Next review page", "N",
                                         self.next_review)
        self.act_replan = self._act("&Replan page", "R", self.replan)
        self.act_fit = self._act("&Fit", "Ctrl+0", self.fit)
        self.act_page_tab = self._act("Show &page", "Ctrl+1",
                                     lambda: self.tabs.setCurrentWidget(self.page_tab))
        self.act_sections_tab = self._act("Show &sections && bars", "Ctrl+2",
                                         lambda: self.tabs.setCurrentWidget(self.sections_tab))
        self.act_map_tab = self._act("Show scan &map", "Ctrl+3",
                                    lambda: self.tabs.setCurrentWidget(self.map_tab))
        self.act_ok = self._act("Mark page &OK", "", lambda: self.set_page_status(STATUS_OK))
        self.act_fix = self._act("Mark page needs &fix", "",
                                 lambda: self.set_page_status(STATUS_NEEDS_FIX))
        self.act_gaps = self._act("Confirm all gaps", "", self.confirm_all_gaps)

    def _make_ui(self):
        self.view = PageView()
        self.header = QLabel("Open a plan-v2 directory")
        self.header.setStyleSheet("QLabel { padding: 4px 8px; color: #ddd; "
                                  "background: #242424; }")
        self.header.setWordWrap(True)
        self.page_tab = QWidget()
        lay = QVBoxLayout(self.page_tab)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.header)
        lay.addWidget(self.view, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.page_tab, "Page")
        self.setCentralWidget(self.tabs)

        # page list
        self.page_list = QListWidget()
        self.page_list.currentRowChanged.connect(self._on_row_changed)
        self.filter = QComboBox()
        for label, value in (("All pages", "all"),
                             ("Needs review", "review"),
                             ("Has gaps", "gaps"),
                             ("Auto bars", "auto"),
                             ("Unreviewed", "unreviewed"),
                             ("Reviewed", "reviewed")):
            self.filter.addItem(label, value)
        self.filter.currentIndexChanged.connect(lambda _=0: self._build_list(True))
        self.count_label = QLabel("")
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.addWidget(self.filter)
        ll.addWidget(self.page_list, 1)
        ll.addWidget(self.count_label)
        self.left_dock = QDockWidget("Pages", self)
        self.left_dock.setObjectName("pages")
        self.left_dock.setWidget(left)
        self.left_dock.setMinimumWidth(260)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)

        # page info
        self.page_title = QLabel("No page")
        self.page_title.setWordWrap(True)
        self.page_title.setStyleSheet("font-weight: bold;")
        self.page_flags = QLabel("")
        self.page_flags.setWordWrap(True)
        self.status_combo = QComboBox()
        for key, label in PAGE_STATUSES.items():
            self.status_combo.addItem(label, key)
        self.status_combo.currentIndexChanged.connect(self._on_status_changed)

        # sections
        self.section_table = QTableWidget(0, 6)
        self.section_table.setHorizontalHeaderLabels(
            ["Keep", "Scan", "Sec", "Rows", "Film", "Bar source"])
        self.section_table.verticalHeader().setVisible(False)
        self.section_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.section_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.section_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.section_table.horizontalHeader().setStretchLastSection(True)
        self.section_table.itemChanged.connect(self._on_section_changed)
        self.section_table.cellDoubleClicked.connect(
            self._on_section_double_clicked)

        # gaps
        self.gap_table = QTableWidget(0, 2)
        self.gap_table.setHorizontalHeaderLabels(["Gap film range", "Status"])
        self.gap_table.verticalHeader().setVisible(False)
        self.gap_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.gap_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.gap_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.gap_table.horizontalHeader().setStretchLastSection(True)
        self.gap_table.setMaximumHeight(160)

        self.note = QPlainTextEdit()
        self.note.setPlaceholderText("Page note")
        self.note.setMaximumHeight(80)

        buttons = QHBoxLayout()
        for text, slot in (("OK", lambda: self.set_page_status(STATUS_OK)),
                           ("Needs fix", lambda: self.set_page_status(STATUS_NEEDS_FIX)),
                           ("Confirm gaps", self.confirm_all_gaps),
                           ("Replan", self.replan),
                           ("Save", self.save)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)
        rl.addWidget(self.page_title)
        rl.addWidget(self.page_flags)
        rl.addWidget(QLabel("Status"))
        rl.addWidget(self.status_combo)
        rl.addWidget(QLabel("Sections (uncheck to drop from the page)"))
        rl.addWidget(self.section_table, 2)
        rl.addWidget(QLabel("Gaps"))
        rl.addWidget(self.gap_table, 1)
        rl.addWidget(QLabel("Note"))
        rl.addWidget(self.note)
        rl.addLayout(buttons)
        rl.addWidget(QLabel("Fix bars in the Sections & bars tab, then press Replan."))
        self.right_dock = QDockWidget("Review", self)
        self.right_dock.setObjectName("review")
        self.right_dock.setWidget(right)
        self.right_dock.setMinimumWidth(520)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)

        # Components: full contributing scans with their bars, plus a
        # highlighted band for the section this page uses.
        self.component_list = QListWidget()
        self.component_list.setIconSize(QSize(180, 55))
        self.component_list.currentRowChanged.connect(self._on_component_row)
        self.component_view = ba.BarView()
        self.component_view.barDrawn.connect(self._on_component_bar_drawn)
        self.component_view.barAdjusted.connect(self._on_component_bar_adjusted)
        self.component_view.autoAdjusted.connect(self._on_component_auto_adjusted)
        self.component_view.autoAdopted.connect(self._on_component_auto_adopted)
        self.component_view.barSelected.connect(self._on_component_bar_selected)
        self.component_view.autoSelected.connect(self._on_component_auto_selected)
        self.component_info = QLabel("Select a page to explode its components")
        self.component_info.setWordWrap(True)
        comp_left = QWidget()
        cl = QVBoxLayout(comp_left)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.addWidget(QLabel("Contributing sections"))
        cl.addWidget(self.component_list, 1)
        cl.addWidget(self.component_info)
        comp_buttons = QHBoxLayout()
        for text, slot in (("Accept auto", self._accept_component_auto),
                           ("Clear bars", self._clear_component_bars),
                           ("Delete bar", self._delete_component_bar),
                           ("Save bars", self._save_bars)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            comp_buttons.addWidget(b)
        cl.addLayout(comp_buttons)
        comp_split = QSplitter(Qt.Horizontal)
        comp_split.addWidget(comp_left)
        comp_split.addWidget(self.component_view)
        comp_split.setSizes([300, 1000])
        self.sections_tab = QWidget()
        sl = QVBoxLayout(self.sections_tab)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(comp_split)
        self.tabs.addTab(self.sections_tab, "Sections & bars")

        # Section map: the page's consecutive scans side by side, with lines
        # linking the sections that cover the same film rows.
        self.map_label = QLabel("Select a page to map its sections")
        self.map_label.setWordWrap(True)
        self.map_label.setStyleSheet("QLabel { padding: 4px 8px; color: #ddd; "
                                     "background: #242424; }")
        self.map_view = SectionMapView()
        self.map_view.linkDropped.connect(self._on_map_link)
        pair_bar = QHBoxLayout()
        self.map_prev = QPushButton("◀ Previous pair")
        self.map_prev.clicked.connect(lambda: self._map_step(-1))
        self.map_next = QPushButton("Next pair ▶")
        self.map_next.clicked.connect(lambda: self._map_step(1))
        self.map_clear = QPushButton("Clear pair links")
        self.map_clear.clicked.connect(self._clear_map_links)
        pair_bar.addWidget(self.map_prev)
        pair_bar.addWidget(self.map_next)
        pair_bar.addWidget(self.map_clear)
        pair_bar.addStretch(1)
        self.map_tab = QWidget()
        ml = QVBoxLayout(self.map_tab)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        ml.addWidget(self.map_label)
        ml.addLayout(pair_bar)
        ml.addWidget(self.map_view, 1)
        self.tabs.addTab(self.map_tab, "Scan map")

        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        for a in (self.act_open, self.act_save, None, self.act_prev,
                  self.act_next, self.act_next_review, None,
                  self.act_replan, self.act_ok, self.act_fix,
                  self.act_gaps, None, self.act_fit, None,
                  self.act_page_tab, self.act_sections_tab,
                  self.act_map_tab):
            tb.addSeparator() if a is None else tb.addAction(a)

        sb = QStatusBar()
        self.status = QLabel("")
        sb.addWidget(self.status)
        self.setStatusBar(sb)

    # ---------- opening -------------------------------------------------

    def open_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Open a plan-v2 directory", str(DEFAULT_PLAN_ROOT))
        if d:
            self.load_plan_dir(Path(d))

    def load_plan_dir(self, plan_dir: Path, overrides: Path | None = None):
        plan_dir = Path(plan_dir).resolve()
        plans = load_plans(plan_dir)
        if not plans:
            QMessageBox.warning(self, "No plans",
                                f"No plan-v2 TOMLs found in\n{plan_dir}")
            return
        self.plan_dir = plan_dir
        self.plans = plans
        self.overrides = OverrideStore(
            overrides or (plan_dir / "section_overrides.json"))
        self.overrides.load()
        if any(p.review.get("needs_review") for p in self.plans):
            self.filter.setCurrentIndex(1)      # Needs review
        self._build_list(False)
        target = next((i for i, p in enumerate(self.plans)
                       if p.review.get("needs_review")), 0)
        self._select_index(target)
        self.setWindowTitle(f"Section / Gap Reviewer — {plan_dir.name}")

    # ---------- list ----------------------------------------------------

    def _page_status(self, page: PagePlan) -> str:
        return self.overrides.get_page(page.label).get("status", STATUS_UNREVIEWED)

    def _reviewed(self, page: PagePlan) -> bool:
        return self._page_status(page) in (STATUS_OK, STATUS_GAP_ACCEPTED,
                                           STATUS_SKIP)

    def _matches(self, page: PagePlan) -> bool:
        want = self.filter.currentData()
        if want == "all":
            return True
        if want == "review":
            return bool(page.review.get("needs_review")) and not self._reviewed(page)
        if want == "gaps":
            return bool(page.gaps)
        if want == "auto":
            return bool(page.review.get("auto_bars"))
        if want == "unreviewed":
            return self._page_status(page) == STATUS_UNREVIEWED
        if want == "reviewed":
            return self._reviewed(page)
        return True

    def _build_list(self, keep_current: bool):
        selected = self.current if keep_current else None
        self.page_list.blockSignals(True)
        self.page_list.clear()
        self.visible = []
        for i, page in enumerate(self.plans):
            if not self._matches(page):
                continue
            status = self._page_status(page)
            flags = []
            if page.review.get("needs_review"):
                flags.append("REVIEW")
            if page.gaps:
                flags.append(f"{len(page.gaps)} gap")
            if page.review.get("auto_bars"):
                flags.append("auto")
            text = f"{page.page_index:04d}  {page.label}"
            if flags:
                text += "  [" + ", ".join(flags) + "]"
            if status != STATUS_UNREVIEWED:
                text += f"  ({PAGE_STATUSES.get(status, status)})"
            self.page_list.addItem(text)
            item = self.page_list.item(self.page_list.count() - 1)
            item.setToolTip(str(page.path))
            if status == STATUS_OK:
                item.setForeground(QColor("#79d99a"))
            elif status == STATUS_NEEDS_FIX:
                item.setForeground(QColor("#ff8a80"))
            elif status == STATUS_GAP_ACCEPTED:
                item.setForeground(QColor("#ffb340"))
            self.visible.append(i)
        self.page_list.blockSignals(False)
        self.count_label.setText(f"{len(self.visible)} / {len(self.plans)} pages")
        if not self.visible:
            self.current = None
            self._refresh()
            return
        row = 0
        if selected is not None and selected in self.visible:
            row = self.visible.index(selected)
        self.page_list.blockSignals(True)
        self.page_list.setCurrentRow(row)
        self.page_list.blockSignals(False)
        if self.current != self.visible[row]:
            self._on_row_changed(row)

    def _select_index(self, index: int):
        if index not in self.visible:
            if not self.visible:
                return
            index = self.visible[0]
        self.page_list.setCurrentRow(self.visible.index(index))

    def _on_row_changed(self, row: int):
        if 0 <= row < len(self.visible):
            self.load_page(self.visible[row])

    def step(self, delta: int):
        if not self.visible:
            return
        row = max(0, min(len(self.visible) - 1,
                         self.page_list.currentRow() + delta))
        self.page_list.setCurrentRow(row)

    def next_review(self):
        if not self.plans:
            return
        start = self.current if self.current is not None else -1
        for k in range(1, len(self.plans) + 1):
            i = (start + k) % len(self.plans)
            page = self.plans[i]
            if page.review.get("needs_review") and not self._reviewed(page):
                if self.filter.currentData() != "all":
                    self.filter.setCurrentIndex(0)
                self._select_index(i)
                return

    # ---------- page ----------------------------------------------------

    def _refresh(self):
        self.page_title.setText("No page")
        self.page_flags.setText("")
        self.section_table.setRowCount(0)
        self.gap_table.setRowCount(0)
        self.note.clear()
        self.header.setText("No page")
        self.view.clear()
        self.map_panels = []
        self.map_pair = 0
        self.map_page_label = None
        self.map_label.setText("Select a page to map its sections")
        self.map_view.clear()

    def load_page(self, index: int):
        if not (0 <= index < len(self.plans)):
            return
        self._commit_note()
        self.current = index
        page = self.plans[index]
        self._loading = True
        try:
            self.page_title.setText(page.label)
            self.page_title.setToolTip(str(page.path))
            flags = []
            if page.review.get("needs_review"):
                flags.append("needs review: " + ", ".join(
                    page.review.get("reasons") or ["structural"]))
            if page.review.get("auto_bars"):
                flags.append("uses auto bars")
            self.page_flags.setText(" · ".join(flags) or "clean")
            rec = self.overrides.get_page(page.label)
            idx = self.status_combo.findData(rec.get("status", STATUS_UNREVIEWED))
            self.status_combo.setCurrentIndex(max(0, idx))
            self.note.setPlainText(rec.get("note") or "")
            self._fill_tables(page, rec)
        finally:
            self._loading = False
        self.recompose()
        self._populate_components(page)
        self._populate_map(page)

    def _fill_tables(self, page: PagePlan, rec: dict):
        self.section_table.blockSignals(True)
        self.section_table.setRowCount(len(page.sections))
        for i, sec in enumerate(page.sections):
            key = page.section_key(sec)
            keep = self.overrides.get_section(page.label, key).get("keep", True)
            item = QTableWidgetItem("")
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if keep else Qt.Unchecked)
            self.section_table.setItem(i, 0, item)
            vals = [f"{sec.get('index')}",
                    f"{sec.get('section')}",
                    f"{sec.get('rows')}",
                    f"{sec.get('film')}",
                    sec.get("bar_source", "")]
            for col, v in enumerate(vals, 1):
                cell = QTableWidgetItem(str(v))
                self.section_table.setItem(i, col, cell)
        self.section_table.blockSignals(False)

        self.gap_table.setRowCount(len(page.gaps))
        for i, gap in enumerate(page.gaps):
            key = PagePlan.gap_key(gap)
            state = self.overrides.get_gap(page.label, key).get(
                "status", GAP_UNREVIEWED)
            self.gap_table.setItem(i, 0, QTableWidgetItem(
                f"{gap[0]:.1f} – {gap[1]:.1f}"))
            combo = QComboBox()
            for k, label in GAP_STATUSES.items():
                combo.addItem(label, k)
            idx = combo.findData(state)
            combo.setCurrentIndex(max(0, idx))
            combo.currentIndexChanged.connect(
                lambda _idx=0, gkey=key, c=combo: self._on_gap_changed(gkey, c))
            self.gap_table.setCellWidget(i, 1, combo)

    def recompose(self):
        if self.current is None:
            return
        page = self.plans[self.current]
        kept = []
        for sec in page.sections:
            key = page.section_key(sec)
            if self.overrides.get_section(page.label, key).get("keep", True):
                kept.append(sec)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            image = self._compose(page, kept)
        finally:
            QApplication.restoreOverrideCursor()
        section_state = {
            page.section_key(s): self.overrides.get_section(
                page.label, page.section_key(s)).get("keep", True)
            for s in page.sections
        }
        gap_state = {
            PagePlan.gap_key(g): self.overrides.get_gap(
                page.label, PagePlan.gap_key(g)).get("status", GAP_UNREVIEWED)
            for g in page.gaps
        }
        self.view.set_image(image, page, page.sections, section_state,
                            page.gaps, gap_state)
        self.header.setText(
            f"{page.label} · {page.film[1] - page.film[0]:.0f} px tall · "
            f"{len(kept)}/{len(page.sections)} sections kept · "
            f"{len(page.gaps)} gap(s)")
        self.status.setText(
            f"{self.current + 1}/{len(self.plans)} · {page.path.name}")

    def _compose(self, page: PagePlan, sections: list[dict]):
        contribs = []
        for sec in sections:
            scan = page.source_dir / sec["scan"]
            try:
                with Image.open(scan) as im:
                    w, h = im.size
            except OSError:
                continue
            r0, r1 = sec["rows"]
            contribs.append({
                "scan": sec["scan"],
                "path": str(scan),
                "bbox": [int(r0), int(r1), 0, w - 1],
                "size": [h, w],
                "film_offset": float(sec["film_offset"]),
                "dx": float(sec["dx"]),
            })
        if not contribs:
            return None
        img, _names = cp.assemble_page(tuple(page.film), contribs, self.lru)
        return img

    # ---------- replan --------------------------------------------------

    def _replan_window(self, page: PagePlan) -> list[str]:
        """A contiguous scan window around the page's contributors.

        `plan_v2` walks pages from bars across a whole run; planning only the
        page's own scans would cut off the boundary bars, so include a few
        neighbours on either side.  The window is what keeps a replan quick
        (bar detection/OCR runs on a handful of scans, not the whole roll).
        """
        names = {s["scan"] for s in page.sections}
        items = pv2.discover_scans(page.source_dir)
        positions = [i for i, item in enumerate(items)
                     if item.path.name in names]
        if not positions:
            return sorted(names)
        lo = max(0, min(positions) - 8)
        hi = min(len(items), max(positions) + 9)
        window = [item.path.name for item in items[lo:hi]]
        # Defensive: never drop one of the page's actual contributors.
        for name in sorted(names):
            if name not in window:
                window.append(name)
        return window

    @classmethod
    def _replan_deltas(cls, old: PagePlan, new: PagePlan) -> list[float]:
        """Candidate film-coordinate shifts for a window-plan page.

        A windowed plan starts its film axis at the window's first scan, while
        the page's existing TOML is on the whole-roll axis.  A matching
        scan+section gives the best shift, but bars can change the section
        number or a pair offset can differ from the old plan, so keep every
        shared-scan shift as a candidate and let the caller pick the one that
        best overlaps the current page's film range.
        """
        deltas = {0.0}
        old_sections = {(str(sec["scan"]), int(sec.get("section", 0))): sec
                        for sec in old.sections}
        old_offsets = {str(sec["scan"]): float(sec["film_offset"])
                       for sec in old.sections}
        for sec in new.sections:
            scan = str(sec["scan"])
            key = (scan, int(sec.get("section", 0)))
            before = old_sections.get(key)
            if before is not None:
                rows = tuple(sec.get("rows") or [0, 0])
                before_rows = tuple(before.get("rows") or [0, 0])
                if cls._rows_similar(before_rows, rows):
                    deltas.add(float(before["film_offset"])
                               - float(sec["film_offset"]))
            if scan in old_offsets:
                deltas.add(old_offsets[scan] - float(sec["film_offset"]))
        return sorted(deltas)

    @staticmethod
    def _best_replan_page(old: PagePlan, plans: list[PagePlan]):
        if not plans:
            return None, 0.0
        old_scans = {s["scan"] for s in old.sections}
        best = None
        for new in plans:
            common = len(old_scans & {s["scan"] for s in new.sections})
            for delta in MainWindow._replan_deltas(old, new):
                overlap = (min(old.film[1], new.film[1] + delta)
                           - max(old.film[0], new.film[0] + delta))
                score = (overlap, common)
                if best is None or score > best[0]:
                    best = (score, new, delta)
        assert best is not None
        overlap, _common = best[0]
        if overlap <= 0:
            return None, 0.0
        return best[1], best[2]

    def _select_label(self, label: str):
        idx = next((i for i, page in enumerate(self.plans)
                    if page.label == label), None)
        if idx is None:
            return
        if idx not in self.visible:
            self.filter.blockSignals(True)
            self.filter.setCurrentIndex(0)
            self.filter.blockSignals(False)
            self._build_list(False)
        self._select_index(idx)

    @staticmethod
    def _rows_similar(a, b) -> bool:
        lo, hi = max(int(a[0]), int(b[0])), min(int(a[1]), int(b[1]))
        if hi < lo:
            return False
        shorter = min(int(a[1]) - int(a[0]) + 1,
                      int(b[1]) - int(b[0]) + 1)
        return (hi - lo + 1) >= 0.9 * max(1, shorter)

    @classmethod
    def _page_subsumed(cls, other: PagePlan, data: dict) -> bool:
        """True when `data` now covers/duplicates this existing page.

        Local replanning can merge a short/gap page into its neighbour.  The
        old page must then disappear, otherwise it lingers with a gap even
        though the new page links to the next scan.  Film coordinates alone
        are not enough (they restart per run), so require a shared
        scan+section and compare the rows.
        """
        if not other.sections:
            return False
        new_secs = {(str(s.get("scan")), int(s.get("section", 0))):
                    tuple(map(int, s.get("rows") or [0, 0]))
                    for s in data.get("sections") or []}
        shared = False
        same_content = True
        contained = True
        for sec in other.sections:
            key = (str(sec.get("scan")), int(sec.get("section", 0)))
            rows = tuple(sec.get("rows") or [0, 0])
            new_rows = new_secs.get(key)
            if new_rows is None:
                same_content = False
                contained = False
                continue
            shared = True
            if not cls._rows_similar(rows, new_rows):
                same_content = False
            if not (new_rows[0] <= rows[0] and new_rows[1] >= rows[1]):
                contained = False
        if not shared:
            return False
        overlap = (min(float(other.film[1]), float(data["film"][1]))
                   - max(float(other.film[0]), float(data["film"][0])))
        height = max(1.0, float(other.film[1]) - float(other.film[0]))
        if overlap / height >= 0.5:
            return True
        if same_content and overlap >= 100.0:
            return True
        if same_content and contained:
            gap = (max(float(other.film[0]), float(data["film"][0]))
                   - min(float(other.film[1]), float(data["film"][1])))
            if overlap > 0 or gap <= 500.0:
                return True
        return False

    def _window_delta(self, page: PagePlan,
                      new_pages: list[PagePlan]) -> float | None:
        """Shift the whole window onto the existing film axis.

        Anchor on the earliest window scan that already has a film offset in
        the plan tree.  That preserves the layout of the run before the
        changed bar / pair, so only the pages at and after the change move.
        """
        if not page.sections:
            return None
        prefix = page.sections[0]["scan"]
        prefix = prefix[:prefix.rfind("_")]
        known = {}
        for other in self.plans:
            for sec in other.sections:
                name = str(sec["scan"])
                if name[:name.rfind("_")] != prefix:
                    continue
                known.setdefault(name, float(sec["film_offset"]))
        anchors = []
        for new in new_pages:
            for sec in new.sections:
                name = str(sec["scan"])
                if name not in known:
                    continue
                idx = scan_index(Path(name))
                if idx is None:
                    continue
                anchors.append((idx, known[name] - float(sec["film_offset"])))
        if not anchors:
            return None
        return min(anchors, key=lambda item: item[0])[1]

    @staticmethod
    def _shift_page_data(new: PagePlan, delta: float, label: str,
                         page_index: int, reverse: bool,
                         source_dir: Path) -> dict:
        sections = []
        for sec in new.sections:
            shifted = dict(sec)
            shifted["film"] = [float(sec["film"][0]) + delta,
                               float(sec["film"][1]) + delta]
            shifted["film_offset"] = float(sec["film_offset"]) + delta
            sections.append(shifted)
        return {
            "label": label,
            "kind": new.kind,
            "page_index": page_index,
            "source_dir": str(source_dir),
            "reverse": reverse,
            "film": [float(new.film[0]) + delta,
                     float(new.film[1]) + delta],
            "sections": sections,
            "gaps": [[float(g[0]) + delta, float(g[1]) + delta]
                     for g in new.gaps],
            "review": new.review,
        }

    def _sync_window(self, page: PagePlan, new_pages: list[PagePlan],
                     names: list[str], delta: float):
        """Re-plan the neighbouring pages a changed pair/bar already moved.

        Only pages whose scans are comfortably inside the planning window are
        rewritten; the scans at each edge exist to give `plan_v2` boundary
        bars, not to replace half-planned edge pages.  Existing page labels and
        numbers are kept, and several old pages that collapse into one new
        page are grouped (lowest page number survives).
        """
        safe = set(names[2:-2]) if len(names) > 4 else set(names)
        candidates = [p for p in self.plans if p.sections
                      and all(s["scan"] in safe for s in p.sections)]
        data_pages = [self._shift_page_data(
            new, delta, new.label, new.page_index, page.reverse,
            page.source_dir) for new in new_pages]
        new_scans = {str(s["scan"]) for data in data_pages
                     for s in data["sections"]}

        # A page containing a scan that the new window plan dropped is an
        # obsolete fragment of a duplicate exposure/stall.  It must not be
        # matched to some later page by film overlap; it is removed below.
        obsolete = [p for p in candidates
                    if {str(s["scan"]) for s in p.sections} - new_scans]
        live = [p for p in candidates if p not in obsolete]

        matches: dict[int, list[PagePlan]] = {}
        for cand in live:
            cand_scans = {s["scan"] for s in cand.sections}
            best = None
            for i, data in enumerate(data_pages):
                new_by_key = {(str(s["scan"]), int(s.get("section", 0))):
                              tuple(s.get("rows") or [0, 0])
                              for s in data["sections"]}
                content = 0
                for sec in cand.sections:
                    key = (str(sec["scan"]), int(sec.get("section", 0)))
                    rows = new_by_key.get(key)
                    if rows is not None and self._rows_similar(
                            tuple(sec.get("rows") or [0, 0]), rows):
                        content += 1
                common = len(cand_scans & {s["scan"]
                                           for s in data["sections"]})
                overlap = (min(float(cand.film[1]), float(data["film"][1]))
                           - max(float(cand.film[0]), float(data["film"][0])))
                if overlap <= 0 and common == 0:
                    continue
                # Prefer the page holding the same section content, even if
                # a neighbouring page happens to overlap the film more.
                score = (content, overlap, common)
                if best is None or score > best[0]:
                    best = (score, i)
            if best is not None:
                matches.setdefault(best[1], []).append(cand)

        keepers: dict[int, PagePlan] = {}
        removed: list[PagePlan] = []
        current_keeper = None
        for i, cands in matches.items():
            keeper = min(cands, key=lambda p: (p.page_index, p.label))
            data = data_pages[i]
            data["label"] = keeper.label
            data["page_index"] = keeper.page_index
            data["reverse"] = keeper.reverse
            pv2.write_plan(keeper.path, data)
            keepers[i] = keeper
            for old in cands:
                if old.path == keeper.path:
                    continue
                self._merge_override(old.label, keeper.label)
                try:
                    old.path.unlink()
                except OSError:
                    pass
                removed.append(old)
            if any(p.label == page.label for p in cands):
                current_keeper = keeper

        # Remove the obsolete fragments, merging their override into the
        # nearest surviving page.  Only remove ones inside the newly planned
        # film range (edge pages may simply be outside the window).
        if keepers and data_pages:
            new_lo = min(float(d["film"][0]) for d in data_pages)
            new_hi = max(float(d["film"][1]) for d in data_pages)
            for cand in obsolete:
                if not (float(cand.film[0]) < new_hi
                        and float(cand.film[1]) > new_lo):
                    continue
                nearest = min(keepers.values(),
                              key=lambda p: abs(p.page_index
                                                - cand.page_index))
                self._merge_override(cand.label, nearest.label)
                try:
                    cand.path.unlink()
                except OSError:
                    pass
                removed.append(cand)

        if current_keeper is None and keepers and any(
                p.label == page.label for p in removed):
            current_keeper = min(keepers.values(),
                                 key=lambda p: abs(p.page_index
                                                   - page.page_index))
        if current_keeper is None:
            return None, removed
        return current_keeper, removed

    def _merge_override(self, src: str, dst: str):
        if self.overrides is None:
            return
        rec = self.overrides.pages.pop(src, None)
        if not rec:
            return
        target = self.overrides.page(dst)
        if target.get("status", STATUS_UNREVIEWED) == STATUS_UNREVIEWED:
            target["status"] = rec.get("status", STATUS_UNREVIEWED)
        for key, val in (rec.get("sections") or {}).items():
            target.setdefault("sections", {}).setdefault(key, val)
        for key, val in (rec.get("gaps") or {}).items():
            target.setdefault("gaps", {}).setdefault(key, val)
        note = (rec.get("note") or "").strip()
        if note and note not in (target.get("note") or ""):
            old = (target.get("note") or "").strip()
            target["note"] = (old + "\n" + note).strip() if old else note
        if (rec.get("updated") or "") > (target.get("updated") or ""):
            target["updated"] = rec["updated"]

    def replan(self):
        """Re-run plan_v2 for a window around this page and load the result.

        The window plan is anchored to the earliest scan that already has a
        film offset, then every neighbouring page comfortably inside the window
        is rewritten too.  A changed pair offset shifts the whole run after it,
        so updating only the selected page would leave its neighbours on a
        different film axis.  Existing page labels/numbers are kept; stale
        short/gap pages subsumed by a new page are absorbed (lowest page number
        survives, duplicate TOML deleted, override merged), and pages whose
        scan was dropped as a duplicate exposure/stall are removed.  The default
        alignment (manual > old plan > OCR) is tried first; if it no longer
        produces a page overlapping this film range, old-plan alignment is
        tried so an unrelated OCR re-detection cannot erase the page.
        """
        if self.current is None:
            return
        page = self.plans[self.current]
        if not page.sections:
            QMessageBox.information(
                self, "Nothing to replan",
                "This page has no contributing sections.")
            return
        self._commit_note()
        if self.overrides is not None:
            self.overrides.save()

        names = self._replan_window(page)
        ann_path = self.annotations_path or default_annotation_path(
            page.source_dir)
        direction = "reverse" if page.reverse else "forward"
        self.statusBar().showMessage(
            f"Replanning {len(names)} scan(s) around {page.label}…")
        chosen = None
        fallback = None
        messages = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for align in ("auto", "plan"):
                output = io.StringIO()
                with tempfile.TemporaryDirectory(
                        prefix="section-replan-") as tmp:
                    argv = [str(page.source_dir),
                            "--output-dir", tmp,
                            "--annotations", str(ann_path),
                            "--section-overrides", str(self.overrides.path),
                            "--plan-root", str(self.plan_root),
                            "--scan-names", ",".join(names),
                            "--direction", direction,
                            "--align", align]
                    try:
                        with contextlib.redirect_stdout(output), \
                                contextlib.redirect_stderr(output):
                            pv2.main(argv)
                    except SystemExit as exc:
                        if exc.code not in (0, None):
                            messages.append(str(exc) or output.getvalue())
                            continue
                    except Exception as exc:  # noqa: BLE001
                        messages.append(f"{type(exc).__name__}: {exc}")
                        continue
                    all_new = load_plans(Path(tmp))
                    dup_records = []
                    summary_path = Path(tmp) / "plan_v2_summary.json"
                    if summary_path.is_file():
                        try:
                            summary = json.loads(summary_path.read_text(
                                encoding="utf-8"))
                            dup_records = list(summary.get("duplicates") or [])
                        except (OSError, json.JSONDecodeError):
                            pass
                    new, delta = self._best_replan_page(page, all_new)
                    if new is not None:
                        chosen = (new, delta, all_new, dup_records)
                        break
                    if fallback is None and all_new:
                        fallback = (all_new, delta, dup_records)
                    messages.append(
                        f"align={align}: no page overlapping the current "
                        f"film range")
        finally:
            QApplication.restoreOverrideCursor()

        if chosen is not None:
            new, delta, all_new, dup_records = chosen
        elif fallback is not None:
            # No new page overlaps this one, but the window plan may show
            # that this page is an obsolete fragment of a dropped duplicate
            # exposure.  Sync the window and remove it if so.
            all_new, delta, dup_records = fallback
            new = None
        else:
            detail = "\n".join(m.strip() for m in messages if m.strip())
            QMessageBox.warning(
                self, "Replan failed",
                "The replan produced no page overlapping this page's film "
                "range." + ("\n\n" + detail if detail else ""))
            self.statusBar().showMessage("Replan failed", 8000)
            return

        window_delta = self._window_delta(page, all_new)
        if window_delta is None:
            window_delta = delta

        keeper, removed = self._sync_window(
            page, all_new, names, window_delta)
        if keeper is None and new is not None:
            # Fallback: the current page sits at a window edge, so just write
            # its own newly planned data as before.
            data = self._shift_page_data(
                new, window_delta, page.label, page.page_index,
                page.reverse, page.source_dir)
            pv2.write_plan(page.path, data)
            keeper = page
            removed = []
        elif keeper is None:
            detail = "\n".join(m.strip() for m in messages if m.strip())
            QMessageBox.warning(
                self, "Replan failed",
                "The replan produced no page overlapping this page's film "
                "range." + ("\n\n" + detail if detail else ""))
            self.statusBar().showMessage("Replan failed", 8000)
            return

        if self.overrides is not None:
            for rec in dup_records:
                scan = str(rec.get("scan") or "")
                if scan:
                    self.overrides.mark_duplicate(
                        scan, str(rec.get("of") or ""),
                        "duplicate exposure (plan_v2)")
        if (removed or dup_records) and self.overrides is not None:
            self.overrides.save()

        label = keeper.label
        self.load_plan_dir(self.plan_dir)
        self._select_label(label)
        extra = ""
        if removed:
            extra = " (absorbed " + ", ".join(
                p.label for p in removed) + ")"
        self.statusBar().showMessage(f"Replanned {label}{extra}", 8000)

    # ---------- components ---------------------------------------------

    def _ensure_bar_context(self, page: PagePlan):
        roll_dir = page.source_dir
        if self.bar_store is not None and self.bar_store.scan_dir == str(roll_dir):
            return
        bar_path = (self.annotations_path
                    or default_annotation_path(roll_dir))
        self.bar_store = AnnotationStore(bar_path)
        self.bar_store.load()
        self.bar_store.scan_dir = str(roll_dir)
        self.ocr_lib = OcrLibrary(roll_dir)
        self.auto_cache = {}

    def _component_key(self, page: PagePlan, sec: dict) -> str:
        name = sec["scan"]
        idx = scan_index(Path(name))
        prefix = name[:name.rfind("_")]
        return f"{page.source_dir.name}/{prefix}/{idx}"

    def _component_auto(self, path: Path, key: str):
        if key not in self.auto_cache:
            ocr = self.ocr_lib.load(path) if self.ocr_lib else None
            self.auto_cache[key] = ba.detect_auto_bars(path, ocr)
        return self.auto_cache[key]

    def _component_icon(self, page: PagePlan, sec: dict):
        path = page.source_dir / sec["scan"]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return QIcon()
        r0, r1 = int(sec["rows"][0]), int(sec["rows"][1])
        crop = img[max(0, r0):min(img.shape[0], r1 + 1), :]
        if crop.size == 0:
            return QIcon()
        h, w = crop.shape
        tw = 180
        th = max(20, int(h * tw / max(1, w)))
        thumb = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        qimg = QImage(thumb.data, thumb.shape[1], thumb.shape[0],
                      thumb.strides[0], QImage.Format_Grayscale8).copy()
        return QIcon(QPixmap.fromImage(qimg))

    def _populate_components(self, page: PagePlan):
        self.component_page = page
        self._ensure_bar_context(page)
        self.component_list.blockSignals(True)
        self.component_list.clear()
        for i, sec in enumerate(page.sections):
            text = (f"{sec.get('index')}s{sec.get('section')}  "
                    f"rows {sec.get('rows')}")
            item = QListWidgetItem(self._component_icon(page, sec), text)
            item.setData(Qt.UserRole, i)
            item.setToolTip(sec.get("scan"))
            self.component_list.addItem(item)
        self.component_list.blockSignals(False)
        if page.sections:
            self.component_list.setCurrentRow(0)
        else:
            self.component_view.clear()
            self.component_info.setText("This page has no components")

    def _scan_offset(self, page: PagePlan, scan: str) -> float | None:
        for sec in page.sections:
            if sec["scan"] == scan:
                return float(sec["film_offset"])
        # Neighbour scans used only for map context are not on this page;
        # any other plan that uses them carries their absolute film offset.
        for other in self.plans:
            for sec in other.sections:
                if sec["scan"] == scan:
                    return float(sec["film_offset"])
        return None

    def _populate_map(self, page: PagePlan):
        if not page.sections:
            self.map_panels = []
            self.map_pair = 0
            self.map_page_label = page.label
            self.map_label.setText(f"{page.label} · no sections to map")
            self.map_view.clear()
            self._update_map_buttons()
            return
        self._ensure_bar_context(page)
        by_scan: dict[str, list[dict]] = {}
        for sec in page.sections:
            by_scan.setdefault(sec["scan"], []).append(sec)

        # Include the contributing scans plus one neighbour on each side so
        # the map shows a genuine consecutive run, not an isolated panel.
        items = pv2.discover_scans(page.source_dir)
        positions = [i for i, item in enumerate(items)
                     if item.path.name in by_scan]
        if positions:
            lo = max(0, min(positions) - 1)
            hi = min(len(items), max(positions) + 2)
            names = [item.path.name for item in items[lo:hi]]
        else:
            names = sorted(by_scan)

        panels = []
        for scan in names:
            offset = self._scan_offset(page, scan)
            if offset is None:
                continue
            secs = by_scan.get(scan, [])
            path = page.source_dir / scan
            key = self._component_key(page, {"scan": scan})
            manual = self.bar_store.get_bars(key) if self.bar_store else []
            auto, bbox = self._component_auto(path, key)
            sections = ba.sections_for(manual or auto, bbox[0], bbox[1])
            used = {}
            for sec in secs:
                number = int(sec.get("section", 0))
                if number:
                    used[number] = self.overrides.get_section(
                        page.label, page.section_key(sec)).get("keep", True)
            panels.append({
                "scan": scan,
                "index": scan_index(path),
                "path": path,
                "offset": offset,
                "sections": sections,
                "used": used,
            })
        same_page = self.map_page_label == page.label
        self.map_page_label = page.label
        self.map_film = page.film
        self.map_panels = panels
        if not same_page or not (0 <= self.map_pair
                                 < max(1, len(panels) - 1)):
            self.map_pair = self._best_map_pair(panels)
        self._render_map_pair()

    def _best_map_pair(self, panels: list[dict]) -> int:
        if len(panels) < 2:
            return 0
        best, best_score = 0, None
        for i in range(len(panels) - 1):
            score = (int(bool(panels[i].get("used")))
                     + int(bool(panels[i + 1].get("used"))),
                     -abs(i - self.map_pair))
            if best_score is None or score > best_score:
                best, best_score = i, score
        return best

    def _map_pair_panels(self) -> list[dict]:
        if len(self.map_panels) < 2:
            return self.map_panels[:1]
        self.map_pair = max(0, min(self.map_pair, len(self.map_panels) - 2))
        return self.map_panels[self.map_pair:self.map_pair + 2]

    def _update_map_buttons(self):
        count = len(self.map_panels)
        has_pair = count >= 2
        self.map_prev.setEnabled(has_pair and self.map_pair > 0)
        self.map_next.setEnabled(has_pair and self.map_pair < count - 2)
        self.map_clear.setEnabled(has_pair)

    def _render_map_pair(self):
        pair = self._map_pair_panels()
        self._update_map_buttons()
        if not pair:
            self.map_label.setText(
                (self.map_page_label or "No page") + " · no sections to map")
            self.map_view.clear()
            return
        count = len(self.map_panels)
        if count >= 2:
            pair_text = f"pair {self.map_pair + 1}/{count - 1}"
        else:
            pair_text = "single scan"
        scans = " ↔ ".join(str(p.get("index", "?")) for p in pair)
        self.map_label.setText(
            f"{self.map_page_label} · {pair_text} · scans {scans} · "
            f"drag S# from one scan onto S# in the other to mark them the "
            f"same (drag again to unlink) · magenta = linked, "
            f"green = page-used, red = dropped")
        self.map_view.set_panels(
            self.map_film, pair, self.overrides.links if self.overrides else {})

    def _map_step(self, delta: int):
        if len(self.map_panels) < 2:
            return
        self.map_pair = max(0, min(self.map_pair + delta,
                                   len(self.map_panels) - 2))
        self._render_map_pair()

    def _on_map_link(self, a: str, b: str):
        if self.overrides is None:
            return
        added = self.overrides.toggle_link(a, b)
        self._mark_dirty()
        self._render_map_pair()
        self.statusBar().showMessage(
            ("Linked " if added else "Unlinked ") + f"{a} ↔ {b}", 4000)

    def _clear_map_links(self):
        if self.overrides is None or len(self.map_panels) < 2:
            return
        left, right = self._map_pair_panels()
        left_keys = {f"{left['scan']}|{n}"
                     for n in range(1, len(left.get("sections") or []) + 1)}
        right_keys = {f"{right['scan']}|{n}"
                      for n in range(1, len(right.get("sections") or []) + 1)}
        if self.overrides.clear_links_between(left_keys, right_keys):
            self._mark_dirty()
            self._render_map_pair()
            self.statusBar().showMessage("Cleared pair links", 4000)

    def _on_component_row(self, row: int):
        if self.component_page is None or row < 0:
            return
        if 0 <= row < len(self.component_page.sections):
            self._load_component(row)

    def _load_component(self, row: int):
        page = self.component_page
        if page is None or not (0 <= row < len(page.sections)):
            return
        sec = page.sections[row]
        path = page.source_dir / sec["scan"]
        key = self._component_key(page, sec)
        manual = self.bar_store.get_bars(key)
        auto, bbox = self._component_auto(path, key)
        # Once manual bars exist they are authoritative, so do not keep
        # showing the dashed amber auto suggestions on top of them.
        shown_auto = [] if manual else auto
        sections = ba.sections_for(manual or auto, bbox[0], bbox[1])
        self.component_view.load(path)
        self.component_view.set_bars(manual, shown_auto, sections, None, None)
        self.component_view.set_highlight(
            sec["rows"], f"PAGE {sec.get('index')}s{sec.get('section')}")
        self.component_section = sec
        self.component_key = key
        self.component_path = path
        self.component_manual = manual
        self.component_auto = list(shown_auto)
        self.component_bbox = bbox
        self.component_selected = None
        self.component_selected_auto = None
        source = "manual bars" if manual else "auto suggestions"
        self.component_info.setText(
            f"{sec['scan']} · section {sec['section']} · "
            f"page rows {sec['rows']} · {source} · plan bar source "
            f"{sec.get('bar_source', 'unknown')}")

    def _refresh_component(self):
        row = self.component_list.currentRow()
        if row >= 0:
            self._load_component(row)
        if self.component_page is not None:
            self._populate_map(self.component_page)

    def _component_mark_fix(self):
        if self.component_page is None:
            return
        rec = self.overrides.page(self.component_page.label)
        if rec.get("status") != STATUS_NEEDS_FIX:
            rec["status"] = STATUS_NEEDS_FIX
            if not rec.get("note"):
                rec["note"] = "bars changed; replan this page (R)"
            self.overrides.touch(self.component_page.label)
            self._mark_dirty()
            idx = self.status_combo.findData(STATUS_NEEDS_FIX)
            self.status_combo.blockSignals(True)
            self.status_combo.setCurrentIndex(max(0, idx))
            self.status_combo.blockSignals(False)

    def _save_bars(self):
        if self.bar_store is not None:
            self.bar_store.save()
            self.statusBar().showMessage(f"Saved {self.bar_store.path}", 4000)

    def _component_bars(self):
        if self.bar_store is None or self.component_key is None:
            return []
        return self.bar_store.get_bars(self.component_key)

    def _on_component_bar_drawn(self, y0: int, y1: int):
        if self.bar_store is None or self.component_key is None:
            return
        self.bar_store.add_bar(self.component_key, y0, y1)
        self._component_mark_fix()
        self._save_bars()
        self._refresh_component()
        self._on_component_bar_selected(len(self._component_bars()) - 1)

    def _on_component_bar_adjusted(self, index: int, y0: int, y1: int):
        if self.bar_store is None or self.component_key is None:
            return
        bars = self._component_bars()
        if not (0 <= index < len(bars)):
            return
        lo, hi = sorted((int(y0), int(y1)))
        if hi - lo < 3:
            return
        bars[index] = [lo, hi]
        self.bar_store.bars[self.component_key] = bars
        self._component_mark_fix()
        self._save_bars()
        self._refresh_component()
        self._on_component_bar_selected(index)

    def _on_component_auto_adjusted(self, index: int, y0: int, y1: int):
        if self.bar_store is None or self.component_key is None:
            return
        if not (0 <= index < len(self.component_auto)):
            return
        auto = list(self.component_auto)
        auto.pop(index)
        self.auto_cache[self.component_key] = (auto, self.component_bbox)
        lo, hi = sorted((int(y0), int(y1)))
        if hi - lo >= 3:
            self.bar_store.add_bar(self.component_key, lo, hi)
        self._component_mark_fix()
        self._save_bars()
        self._refresh_component()
        self._on_component_bar_selected(len(self._component_bars()) - 1)

    def _on_component_auto_adopted(self, index: int):
        if self.bar_store is None or self.component_key is None:
            return
        if not (0 <= index < len(self.component_auto)):
            return
        auto = list(self.component_auto)
        lo, hi = auto.pop(index)
        self.auto_cache[self.component_key] = (auto, self.component_bbox)
        self.bar_store.add_bar(self.component_key, lo, hi)
        self._component_mark_fix()
        self._save_bars()
        self._refresh_component()
        self._on_component_bar_selected(len(self._component_bars()) - 1)

    def _on_component_bar_selected(self, index: int):
        self.component_selected = index
        self.component_selected_auto = None
        if 0 <= index < len(self.component_manual):
            self.component_view.set_selected(index)

    def _on_component_auto_selected(self, index: int):
        self.component_selected_auto = index
        self.component_selected = None
        self.component_view.set_selected(None, index)

    def _delete_component_bar(self):
        if (self.bar_store is None or self.component_key is None
                or self.component_selected is None):
            return
        bars = self._component_bars()
        idx = self.component_selected
        if 0 <= idx < len(bars):
            self.bar_store.remove_bar(self.component_key, idx)
            self.component_selected = None
            self._component_mark_fix()
            self._save_bars()
            self._refresh_component()

    def _accept_component_auto(self):
        if self.bar_store is None or self.component_key is None:
            return
        auto = list(self.component_auto)
        for lo, hi in auto:
            self.bar_store.add_bar(self.component_key, lo, hi)
        self.auto_cache[self.component_key] = ([], self.component_bbox)
        self._component_mark_fix()
        self._save_bars()
        self._refresh_component()

    def _clear_component_bars(self):
        if self.bar_store is None or self.component_key is None:
            return
        self.bar_store.bars.pop(self.component_key, None)
        self.auto_cache.pop(self.component_key, None)
        self.component_selected = None
        self._component_mark_fix()
        self._save_bars()
        self._refresh_component()

    # ---------- edits ---------------------------------------------------

    def _on_status_changed(self, _idx: int):
        if self._loading or self.current is None:
            return
        page = self.plans[self.current]
        rec = self.overrides.page(page.label)
        rec["status"] = self.status_combo.currentData()
        self.overrides.touch(page.label)
        self._mark_dirty()
        self._build_list(True)

    def set_page_status(self, status: str):
        if self.current is None:
            return
        idx = self.status_combo.findData(status)
        self.status_combo.setCurrentIndex(max(0, idx))
        self.statusBar().showMessage(PAGE_STATUSES.get(status, status), 4000)

    def _on_section_changed(self, item: QTableWidgetItem):
        if self._loading or self.current is None or item.column() != 0:
            return
        page = self.plans[self.current]
        row = item.row()
        if not (0 <= row < len(page.sections)):
            return
        key = page.section_key(page.sections[row])
        rec = self.overrides.section(page.label, key)
        rec["keep"] = item.checkState() == Qt.Checked
        self.overrides.touch(page.label)
        self._mark_dirty()
        self.recompose()
        self._populate_map(page)
        self._build_list(True)

    def _on_section_double_clicked(self, row: int, _col: int):
        if self.current is None or not (0 <= row < self.component_list.count()):
            return
        self.component_list.setCurrentRow(row)
        self.tabs.setCurrentWidget(self.sections_tab)

    def _on_gap_changed(self, key: str, combo: QComboBox):
        if self._loading or self.current is None:
            return
        page = self.plans[self.current]
        self.overrides.gap(page.label, key)["status"] = combo.currentData()
        self.overrides.touch(page.label)
        self._mark_dirty()
        self.recompose()

    def confirm_all_gaps(self):
        if self.current is None:
            return
        page = self.plans[self.current]
        for gap in page.gaps:
            self.overrides.gap(page.label, PagePlan.gap_key(gap))["status"] = \
                GAP_CONFIRMED
        self.overrides.page(page.label)["status"] = STATUS_GAP_ACCEPTED
        self.overrides.touch(page.label)
        self._mark_dirty()
        self._build_list(True)
        idx = self.status_combo.findData(STATUS_GAP_ACCEPTED)
        self.status_combo.setCurrentIndex(max(0, idx))
        self.recompose()

    def _commit_note(self):
        if self._loading or self.current is None:
            return
        page = self.plans[self.current]
        text = self.note.toPlainText().strip()
        rec = self.overrides.get_page(page.label)
        if text != (rec.get("note") or ""):
            rec = self.overrides.page(page.label)
            rec["note"] = text
            self.overrides.touch(page.label)
            self._mark_dirty()

    # ---------- persistence --------------------------------------------

    def _mark_dirty(self):
        self._dirty = True
        self._autosave.start()

    def _autosave_now(self):
        if self.overrides is not None:
            try:
                self._commit_note()
                self.overrides.save()
                self._dirty = False
            except OSError as exc:
                self.statusBar().showMessage(f"Save failed: {exc}", 8000)

    def save(self):
        if self.overrides is None:
            return
        self._commit_note()
        self.overrides.save()
        self._dirty = False
        self.statusBar().showMessage(f"Saved {self.overrides.path}", 4000)

    def closeEvent(self, event):  # noqa: N802
        if self.overrides is not None:
            try:
                self._commit_note()
                self.overrides.save()
            except OSError:
                pass
        event.accept()

    def fit(self):
        self.view.fit()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan_dir", nargs="?", help="plan-v2 directory")
    ap.add_argument("--overrides", help="section override JSON path")
    ap.add_argument("--annotations", help="bar/pair annotation JSON path")
    ap.add_argument("--plan-root", default="plans/ocr_1028",
                    help="old plans used by plan_v2 for pair offsets/bars")
    args = ap.parse_args(argv)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Section / Gap Reviewer")
    win = MainWindow(Path(args.plan_dir) if args.plan_dir else None,
                     Path(args.overrides) if args.overrides else None,
                     Path(args.annotations) if args.annotations else None,
                     Path(args.plan_root) if args.plan_root else None)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
