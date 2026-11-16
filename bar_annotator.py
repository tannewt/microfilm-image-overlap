#!/usr/bin/env python3
"""Mark the blank page-boundary bars on each scan.

The page layout in this pipeline is ultimately much simpler than a pair
offset: a scan is split by its blank bars into **sections** (1, 2 or 3
page pieces), and a page is the union of corresponding sections from one or
two adjacent scans.  This viewer exists to get the first half of that right:
mark one or two blank bars per scan, visually check the resulting sections,
and save the bars in the shared pair-annotator JSON.

Bars are stored per scan as ``(row_start, row_end)`` in full-resolution scan
pixels under ``bars`` in ``plans/ocr_1028/annotations/<roll>.json`` -- the
same file the pair annotator uses.  Pair anchors and bars therefore stay in
one place.

Usage:
    python bar_annotator.py 20261028-newspaper-scans/P72-2707-CENLIB9102
    python bar_annotator.py 20261028-newspaper-scans/P72-2707-CENLIB9102 --at 570
    python bar_annotator.py --annotations plans/ocr_1028/annotations/P72-2707-CENLIB9102.json

Controls
--------
    Ctrl+O            open a roll directory
    Ctrl+Shift+O      open an annotation file
    Ctrl+S            save (autosaves too)
    Right / Space     next scan
    Left              previous scan
    N                 next scan with no manual bars
    drag              drag on the scan to add a bar
    drag edge         drag a selected bar's top or bottom edge to adjust it
    double-click      convert an auto bar to a manual one
    Delete            delete the selected manual bar / dismiss an auto suggestion
    A                 accept all auto-detected bars for this scan
    C                 clear this scan's manual bars

The scan always fits the window; there is no zoom or pan.

Auto-detected bars are shown dashed amber as suggestions.  The detector
combines four independent signals:

* dark film bars (the planner's row-projection + column-coverage check),
* white head-to-tail gaps with a finer smoothing window than the planner's
  global pass (real gaps can be only ~90 px),
* OCR-empty row bands (no OCR box covers the rows; must also look bright/
  low-ink or dark), including the longest dark sub-run of a mixed band when
  a page margin and a dark bar merge into a dark repeated exposure, and
* a thick, nearly-horizontal **diagonal dark path** found by dynamic
  programming left-to-right (for bars that drift a few degrees and fail the
  planner's per-column check).

Old-plan ``blank_rows`` are merged in as extra suggestions.  OCR is also a
safety constraint: every selected bar is trimmed away from OCR boxes, and a
tall OCR-only gap without black/diagonal detector support becomes a small
~100 px bar centred between the surrounding OCR boxes, leaving white margins
around the text.  Selection is capped at **one or two bars**: a second is
allowed only when it is at least ~40% of a page away and there are real OCR
boxes between the two bands.  If every detector misses, a fallback band is
offered when a text-free gap exists; when OCR boxes cover the whole content
no bar is suggested rather than one that would cover text.  None of this is
authoritative: manual bars are solid green and are what the plan v2
generator uses.  To turn an auto bar into a manual one, drag its top/bottom
edge (it adjusts as it converts) or double-click it.  The section labels
(S1/S2/S3) update as you mark bars.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
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
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from pair_annotator import (
    AnnotationStore,
    IMAGE_EXTS,
    OcrLibrary,
    default_annotation_path,
    scan_index,
)
import cv2
import numpy as np
import plan_pages as pp

PLAN_ROOT = Path(__file__).resolve().parent / "plans" / "ocr_1028"


@dataclass
class ScanItem:
    path: Path
    prefix: str
    index: int

    def key(self, roll: str) -> str:
        return f"{roll}/{self.prefix}/{self.index}"


def discover_scans(directory: Path) -> list[ScanItem]:
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        idx = scan_index(path)
        if idx is None:
            continue
        groups[path.name[:path.name.rfind("_")]].append((idx, path))
    out: list[ScanItem] = []
    for prefix, entries in sorted(groups.items(),
                                  key=lambda kv: (min(i for i, _ in kv[1]),
                                                  kv[0])):
        for idx, path in sorted(entries):
            out.append(ScanItem(path=path, prefix=prefix, index=idx))
    return out


def merge_bars(bars, gap=8):
    out = []
    for lo, hi in sorted((int(a), int(b)) for a, b in bars if b > a):
        if out and lo <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(b) for b in out]


def sections_for(bars, top, bottom, min_height=20):
    """Section row ranges between the bars, top-to-bottom, 1-based."""
    out = []
    cur = int(top)
    for lo, hi in merge_bars(bars):
        lo = max(int(top), lo)
        hi = min(int(bottom), hi)
        if hi < lo:
            continue
        if lo > cur:
            if lo - cur >= min_height:
                out.append((cur, lo - 1))
        cur = max(cur, hi + 1)
    if bottom >= cur and bottom - cur + 1 >= min_height:
        out.append((cur, int(bottom)))
    return out


def _ocr_empty_bands(gray, bbox, ocr_scan):
    """Page-boundary-sized row bands with no OCR text at all.

    A real gap often carries no words; OCR is a second, independent signal
    that the planner's bar detectors do not use.  Require the band to be
    either bright/low-ink (white junction) or dark (film bar), so a blank
    ad area or an OCR failure inside text is not accepted blindly.
    """
    if ocr_scan is None or len(ocr_scan.entries) < 5:
        # A title card / blank frame has no OCR; don't turn its whole
        # unprinted area into a "bar" suggestion.
        return []
    top, bottom, left, right = bbox
    h = bottom - top + 1
    covered = np.zeros(h, dtype=bool)
    for e in ocr_scan.entries:
        y0 = max(top, int(round(e.rect[1])))
        y1 = min(bottom, int(round(e.rect[3])))
        if y1 >= y0:
            covered[y0 - top:y1 - top + 1] = True
    out = []
    y = 0
    while y < h:
        if covered[y]:
            y += 1
            continue
        j = y
        while j < h and not covered[j]:
            j += 1
        run = j - y
        # Inter-line gaps are ~10-30 px; a boundary is much taller.  A huge
        # run is a blank page or failed OCR, not a boundary.
        if 40 <= run <= 0.35 * h:
            a, b = top + y, top + j - 1
            band = gray[a:b + 1, left:right + 1]
            mean = float(band.mean())
            ink = float((band < 200).mean())
            if (mean > 185 and ink < 0.06) or mean < 110:
                out.append((a, b))
            elif a > top + 50 and (band.mean(axis=1) > 150).sum() >= 10:
                # A mixed OCR-empty run (page margin + dark bar merged into
                # a dark repeated exposure).  Find the longest dark sub-run;
                # this is the 587 case: the black bar runs into the dark
                # repeated page, so the whole run is neither bright nor dark.
                # Runs starting at the content top are page content, not a
                # boundary; skip them.
                rows = np.convolve(band.mean(axis=1), np.ones(11) / 11,
                                   mode="same")
                dark = rows < 120
                best = None
                k = 0
                while k < len(dark):
                    if not dark[k]:
                        k += 1
                        continue
                    m = k
                    while m < len(dark) and dark[m]:
                        m += 1
                    length = m - k
                    if 40 <= length <= min(400, int(0.18 * h)):
                        if best is None or length > best[0]:
                            best = (length, a + k, a + m - 1)
                    k = m
                if best is not None:
                    out.append((best[1], best[2]))
        y = j
    return out


def _diagonal_dark_bands(gray, bbox, max_step=2, dark_thresh=95,
                         min_dark_frac=0.82):
    """Find a thick, nearly-horizontal dark path across the content.

    The row-projection detector assumes a perfectly horizontal bar; a film
    bar that drifts a few degrees diagonally fails its per-column check.
    A small dynamic-programming path from left to right, allowed at most
    `max_step` downscaled rows per column, finds that path.  Thin dark rules
    are rejected by requiring a thick dark run around the path.
    """
    top, bottom, left, right = bbox
    w, h = right - left + 1, bottom - top + 1
    if w < 100 or h < 100:
        return []
    scale = min(1.0, 900.0 / w)
    sw, sh = max(50, int(round(w * scale))), max(50, int(round(h * scale)))
    small = cv2.resize(gray[top:bottom + 1, left:right + 1], (sw, sh),
                       interpolation=cv2.INTER_AREA)
    dark = small < dark_thresh
    # Dark pixel -> low cost; bright -> high cost.
    cost = np.clip((small.astype(np.float32) - 20.0) / 180.0, 0.0, 1.0)
    inf = np.float32(1e9)
    prev = cost[:, 0].copy()
    back = np.zeros((sh, sw), dtype=np.int16)
    for x in range(1, sw):
        best = np.full(sh, inf, dtype=np.float32)
        arg = np.zeros(sh, dtype=np.int16)
        for d in range(-max_step, max_step + 1):
            if d < 0:
                rolled = np.full(sh, inf, dtype=np.float32)
                rolled[:d] = prev[-d:]
            elif d > 0:
                rolled = np.full(sh, inf, dtype=np.float32)
                rolled[d:] = prev[:-d]
            else:
                rolled = prev
            take = rolled < best
            best[take] = rolled[take]
            arg[take] = d
        prev = best + cost[:, x]
        back[:, x] = arg
    y = int(np.argmin(prev))
    path = np.empty(sw, dtype=np.int16)
    path[-1] = y
    for x in range(sw - 1, 0, -1):
        # back[y, x] is the step d chosen for current row y; the previous
        # row is y - d.
        y = int(y - back[y, x])
        path[x - 1] = y
    vals = small[path, np.arange(sw)]
    if float((vals < dark_thresh).mean()) < min_dark_frac:
        return []
    if int(path.max() - path.min()) > 0.08 * sh:
        return []
    # Thickness of the dark run around the path, median over columns.
    thickness = []
    for x in range(sw):
        y0 = int(path[x])
        a = b = y0
        while a > 0 and dark[a - 1, x]:
            a -= 1
        while b + 1 < sh and dark[b + 1, x]:
            b += 1
        thickness.append(b - a + 1)
    thick = max(2, int(np.median(thickness)))
    # A text line is a row of glyphs with white between them; a film bar is a
    # thick solid dark run.  Requiring several downscaled rows rejects text
    # without needing the whole (tilted) bounding band to be dark.
    if thick < 8:
        return []
    s_lo = max(0, int(path.min() - thick / 2.0))
    s_hi = min(sh - 1, int(path.max() + thick / 2.0))
    lo = top + (s_lo - thick / 2.0) / scale
    hi = top + (s_hi + thick / 2.0) / scale
    lo, hi = max(top, int(round(lo))), min(bottom, int(round(hi)))
    return [(lo, hi)] if hi - lo >= 8 else []


def _candidate_bars(gray, bbox, ocr_scan):
    """Raw bar candidates with a rough evidence score."""
    top, bottom, left, right = bbox
    h = bottom - top + 1
    max_thick = max(40, int(0.18 * h))
    out = []

    def add(bars, score, source):
        for lo, hi in bars:
            lo, hi = max(top, int(lo)), min(bottom, int(hi))
            if 12 <= hi - lo <= max_thick:
                out.append((lo, hi, float(score), source))

    add(pp.detect_bars(gray), 3.0, "dark")
    add(pp.detect_white_gaps(gray, bbox, win=81, min_run=40,
                             min_band=40, merge_within=120), 2.0, "white")
    add(_diagonal_dark_bands(gray, bbox), 2.5, "diagonal")
    add(_ocr_empty_bands(gray, bbox, ocr_scan), 1.5, "ocr-empty")
    return out


def _merge_candidates(candidates, gap=20):
    """Collapse overlapping candidates, keeping the strongest evidence."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: (c[0], c[1]))
    groups = []
    for c in ordered:
        if groups and c[0] <= groups[-1][-1][1] + gap:
            groups[-1].append(c)
        else:
            groups.append([c])
    out = []
    for group in groups:
        # Highest score; on a tie prefer the thicker band.
        best = max(group, key=lambda c: (c[2], c[1] - c[0]))
        out.append(best)
    return out


def _ocr_between(ocr_scan, y0, y1):
    if ocr_scan is None or not ocr_scan.entries:
        return 0
    n = 0
    for e in ocr_scan.entries:
        cy = (e.rect[1] + e.rect[3]) / 2.0
        if y0 + 5 < cy < y1 - 5:
            n += 1
    return n


# A bar must never cut through an OCR box.  A two-pixel clearance keeps a
# sliver of white between the bar and every text box; the caller is free to
# compose with more white margin around the OCR.
_OCR_CLEARANCE = 2
_MIN_BAR_HEIGHT = 12

# An OCR-empty run can be much taller than a real film bar: OCR only proves
# that no text lives there, not that the whole blank area is a bar.  When no
# black/diagonal detector backs it up, keep a small marker in the middle of
# the OCR gap instead.
_OCR_ONLY_BAR_HEIGHT = 100
_BLACK_BAR_SOURCES = frozenset({"dark", "diagonal"})


def _ocr_row_boxes(ocr_scan):
    """OCR text row spans, rounded outward so no glyph row is included."""
    if ocr_scan is None:
        return []
    boxes = []
    for e in ocr_scan.entries:
        y0, y1 = e.rect[1], e.rect[3]
        if y1 < y0:
            y0, y1 = y1, y0
        boxes.append((int(np.floor(y0)), int(np.ceil(y1))))
    boxes.sort()
    return boxes


def _ocr_free_runs(lo, hi, boxes, clearance=0):
    """Sub-ranges of ``[lo, hi]`` that stay clear of every OCR row box."""
    if hi < lo:
        return []
    runs = []
    cursor = lo
    for y0, y1 in boxes:
        if y1 + clearance < lo:
            continue
        if y0 - clearance > hi:
            break
        block_lo = max(lo, y0 - clearance)
        block_hi = min(hi, y1 + clearance)
        if block_hi < block_lo:
            continue
        if block_lo > cursor:
            runs.append((cursor, block_lo - 1))
        cursor = max(cursor, block_hi + 1)
        if cursor > hi:
            break
    if cursor <= hi:
        runs.append((cursor, hi))
    return runs


def _longest_run(runs):
    if not runs:
        return None
    return max(runs, key=lambda r: (r[1] - r[0], -r[0]))


def _ocr_safe_bar(lo, hi, boxes):
    """Largest part of ``[lo, hi]`` that does not cover an OCR box."""
    run = _longest_run(_ocr_free_runs(lo, hi, boxes, _OCR_CLEARANCE))
    if run is None or run[1] - run[0] + 1 < _MIN_BAR_HEIGHT:
        return None
    return run


def _small_bar_in(lo, hi, target=_OCR_ONLY_BAR_HEIGHT):
    """A small bar centred inside a larger OCR-free run."""
    size = min(int(target), hi - lo + 1)
    if size < _MIN_BAR_HEIGHT:
        return (lo, hi)
    start = (lo + hi - size + 1) // 2
    start = max(lo, min(start, hi - size + 1))
    return (start, start + size - 1)


def _has_black_evidence(spans, lo, hi, gap=20):
    return any(a <= hi + gap and b >= lo - gap for a, b in spans)


def _normalize_candidates(candidates, ocr_scan):
    """Keep candidates off OCR text and shrink OCR-only oversized bands."""
    boxes = _ocr_row_boxes(ocr_scan)
    if not boxes:
        return candidates
    black = [(lo, hi) for lo, hi, _score, source in candidates
             if source in _BLACK_BAR_SOURCES]
    out = []
    for lo, hi, score, source in candidates:
        safe = _ocr_safe_bar(lo, hi, boxes)
        if safe is None:
            continue
        a, b = safe
        # OCR says "no text here", but only a black detector says "bar".
        # A tall, unbacked OCR gap becomes a small bar in its middle, which
        # leaves white margins around the surrounding OCR text.
        if (source == "ocr-empty"
                and b - a + 1 > _OCR_ONLY_BAR_HEIGHT
                and not _has_black_evidence(black, lo, hi)):
            a, b = _small_bar_in(a, b)
        out.append((a, b, score, source))
    return out


def _fallback_bar(gray, bbox, ocr_scan):
    """A single best-effort suggestion when every detector came up empty.

    With OCR available, only an OCR-free stretch can be used.  If the OCR
    boxes leave no usable gap, no suggestion is returned rather than a bar
    that would cover text.
    """
    top, bottom, left, right = bbox
    h = bottom - top + 1
    boxes = _ocr_row_boxes(ocr_scan)
    if boxes:
        best = _longest_run(_ocr_free_runs(top, bottom, boxes,
                                           _OCR_CLEARANCE))
        if best is not None and best[1] - best[0] + 1 >= 40:
            return [_small_bar_in(best[0], best[1])]
        return []
    # Otherwise pick the lowest-ink 100 px window in the content.
    band = gray[top:bottom + 1, left:right + 1]
    ink = (band < 200).mean(axis=1)
    win = min(120, max(40, h // 20))
    if len(ink) >= win:
        sums = np.convolve(ink, np.ones(win) / win, mode="valid")
        y = int(np.argmin(sums))
        center = top + y + win // 2
    else:
        center = top + h // 2
    return [(max(top, center - 50), min(bottom, center + 50))]


def _choose_bars(candidates, ocr_scan, bbox, gray):
    """At most two bars, far apart, with OCR text between them.

    Bars are trimmed away from OCR boxes before selection.  A fallback is
    returned when every detector missed and a text-free gap exists.  A
    second bar is allowed only when it is at least ~40% of a page away and
    there are real OCR boxes between the two bands.
    """
    top, bottom = bbox[0], bbox[1]
    h = bottom - top + 1
    min_sep = max(600, int(0.40 * h))
    candidates = _normalize_candidates(candidates, ocr_scan)
    ordered = sorted(candidates, key=lambda c: (-c[2], -(c[1] - c[0])))
    if not ordered:
        return _fallback_bar(gray, bbox, ocr_scan)
    first = ordered[0]
    second = None
    for cand in ordered[1:]:
        a, b = sorted((first, cand), key=lambda c: c[0])
        gap = b[0] - a[1]
        if gap < min_sep:
            continue
        if _ocr_between(ocr_scan, a[1], b[0]) >= 3:
            second = cand
            break
    bars = [first] + ([second] if second is not None else [])
    bars.sort(key=lambda c: c[0])
    return [(int(c[0]), int(c[1])) for c in bars]


def detect_auto_bars(path: Path, ocr_scan=None, plan_bars=None):
    """Return one or two bar suggestions.

    Candidates come from dark bars, fine-window white gaps, OCR-empty
    bands, and diagonal dark paths; old-plan ``blank_rows`` are added as
    weaker candidates.  Overlapping candidates merge, then at most two far
    apart candidates are selected, with OCR text required between the two
    bands.  Selected bars are clipped so they never cover an OCR box.  A
    large OCR-only gap without black-detector evidence becomes a small bar
    centred between the surrounding OCR boxes.  If nothing is found, a
    fallback row is suggested when there is a text-free gap.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return [], (0, 0, 0, 0)
    bbox = pp.content_bbox(gray)
    top, bottom = bbox[0], bbox[1]
    cands = _candidate_bars(gray, bbox, ocr_scan)
    for lo, hi in (plan_bars or []):
        lo, hi = max(top, int(lo)), min(bottom, int(hi))
        if 12 <= hi - lo <= max(40, int(0.18 * (bottom - top + 1))):
            cands.append((lo, hi, 2.0, "old-plan"))
    cands = _merge_candidates(cands)
    return _choose_bars(cands, ocr_scan, bbox, gray), bbox


class BarView(QGraphicsView):
    """Fit-to-window scan view with draggable bar top/bottom edges."""

    barDrawn = Signal(int, int)          # new bar y0, y1
    barAdjusted = Signal(int, int, int)  # manual bar index, new y0, new y1
    autoAdjusted = Signal(int, int, int) # auto bar index, new y0, new y1
    autoAdopted = Signal(int)            # convert this auto bar to manual
    barSelected = Signal(int)            # selected manual bar index
    autoSelected = Signal(int)           # selected auto bar index
    cursorMoved = Signal(object)         # QPointF

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing
                            | QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setInteractive(False)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor("#181818")))
        self.viewport().setCursor(Qt.CrossCursor)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._overlays: list[QGraphicsItem] = []
        self._labels: list[QGraphicsItem] = []
        self._image_size = QSize()
        self._manual: list[tuple[int, int]] = []
        self._auto: list[tuple[int, int]] = []
        self._sections: list[tuple[int, int]] = []
        self._selected: int | None = None
        self._selected_auto: int | None = None
        self._create_start: float | None = None
        self._bar_temp: QGraphicsRectItem | None = None
        self._drag: tuple[int, str, float] | None = None
        self._preview: tuple[int, int, int] | None = None
        self._drag_auto: tuple[int, str, float] | None = None
        self._auto_preview: tuple[int, int, int] | None = None
        self._highlight: tuple[int, int, str] | None = None

    # ---------- content ------------------------------------------------

    def load(self, path: Path) -> bool:
        self._scene.clear()
        self._overlays = []
        self._labels = []
        self._pixmap_item = None
        self._create_start = None
        self._bar_temp = None
        self._drag = None
        self._preview = None
        self._drag_auto = None
        self._auto_preview = None
        self._selected = None
        self._selected_auto = None
        self._highlight = None
        img = QImage(str(path))
        if img.isNull():
            return False
        self._image_size = img.size()
        pixmap = QPixmap.fromImage(img)
        item = self._scene.addPixmap(pixmap)
        item.setPos(0, 0)
        self._pixmap_item = item
        self._scene.setSceneRect(QRectF(0, 0, img.width(), img.height()))
        self.fit()
        return True

    def image_size(self) -> QSize:
        return self._image_size

    def fit(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self.viewport().setCursor(Qt.CrossCursor)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.fit()

    def set_bars(self, manual, auto, sections, selected=None,
                 selected_auto=None):
        self._manual = list(manual)
        self._auto = list(auto)
        self._sections = list(sections)
        self._selected = selected
        self._selected_auto = selected_auto
        self._preview = None
        self._auto_preview = None
        self._redraw_overlays()

    def set_selected(self, index, auto_index=None):
        self._selected = index
        self._selected_auto = auto_index
        self._redraw_overlays()

    def set_highlight(self, rows, label="PAGE"):
        self._highlight = (int(rows[0]), int(rows[1]), str(label))
        self._redraw_overlays()

    # ---------- drawing -------------------------------------------------

    def _redraw_overlays(self):
        for item in self._overlays + self._labels:
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._overlays = []
        self._labels = []
        if self._pixmap_item is None:
            return
        w = self._image_size.width()
        if self._highlight is not None:
            h_lo, h_hi, h_label = self._highlight
            rect = QGraphicsRectItem(0, h_lo, w, max(1, h_hi - h_lo))
            rect.setBrush(QBrush(QColor(90, 169, 255, 48)))
            rect.setPen(QPen(QColor("#5aa9ff"), 2, Qt.DashLine))
            rect.setZValue(6)
            self._scene.addItem(rect)
            self._overlays.append(rect)
            text = QGraphicsSimpleTextItem(h_label)
            font = QFont()
            font.setBold(True)
            font.setPointSize(9)
            text.setFont(font)
            text.setBrush(QBrush(QColor("#9fc8ff")))
            text.setPos(10, h_lo + 4)
            text.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            text.setZValue(22)
            self._scene.addItem(text)
            self._overlays.append(text)
        autos = list(self._auto)
        if self._auto_preview is not None:
            i, lo, hi = self._auto_preview
            if 0 <= i < len(autos):
                autos[i] = (lo, hi)
        for i, (lo, hi) in enumerate(autos):
            selected = i == self._selected_auto
            rect = QGraphicsRectItem(0, lo, w, max(1, hi - lo))
            rect.setBrush(QBrush(QColor(255, 179, 64, 60 if selected else 28)))
            rect.setPen(QPen(QColor("#ffd166" if selected else "#ffb340"),
                             3 if selected else 2, Qt.DashLine))
            rect.setZValue(14 if selected else 8)
            self._scene.addItem(rect)
            self._overlays.append(rect)
            if selected:
                for y in (lo, hi):
                    handle = QGraphicsRectItem(-7, -3, 14, 6)
                    handle.setBrush(QBrush(QColor("#ffb340")))
                    handle.setPen(QPen(QColor("#202020"), 1))
                    handle.setPos(28, y)
                    handle.setFlag(QGraphicsItem.ItemIgnoresTransformations,
                                   True)
                    handle.setZValue(28)
                    self._scene.addItem(handle)
                    self._overlays.append(handle)

        bars = list(self._manual)
        if self._preview is not None:
            i, lo, hi = self._preview
            if 0 <= i < len(bars):
                bars[i] = (lo, hi)
        for i, (lo, hi) in enumerate(bars):
            selected = i == self._selected
            rect = QGraphicsRectItem(0, lo, w, max(1, hi - lo))
            rect.setBrush(QBrush(QColor(51, 209, 122, 65 if selected else 45)))
            rect.setPen(QPen(QColor("#ffd166" if selected else "#33d17a"),
                             3 if selected else 2))
            rect.setZValue(12 if selected else 10)
            self._scene.addItem(rect)
            self._overlays.append(rect)
            if selected:
                # Constant-size grab handles at the left edge.
                for y in (lo, hi):
                    handle = QGraphicsRectItem(-7, -3, 14, 6)
                    handle.setBrush(QBrush(QColor("#ffd166")))
                    handle.setPen(QPen(QColor("#202020"), 1))
                    handle.setPos(28, y)
                    handle.setFlag(QGraphicsItem.ItemIgnoresTransformations,
                                   True)
                    handle.setZValue(28)
                    self._scene.addItem(handle)
                    self._overlays.append(handle)

        for n, (lo, hi) in enumerate(self._sections, 1):
            label = QGraphicsSimpleTextItem(f"S{n}")
            font = QFont()
            font.setBold(True)
            font.setPointSize(12)
            label.setFont(font)
            label.setBrush(QBrush(QColor("#ffd166")))
            label.setPos(24, (lo + hi) / 2)
            label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            label.setZValue(25)
            self._scene.addItem(label)
            self._labels.append(label)

    # ---------- hit testing --------------------------------------------

    def _scene_pos(self, event):
        return self.mapToScene(event.position().toPoint())

    def _tol(self) -> float:
        scale = max(1e-6, self.transform().m11())
        return max(3.0, 8.0 / scale)

    def _hit(self, y: float):
        tol = self._tol()
        best = None
        for source, bars in (("manual", self._manual), ("auto", self._auto)):
            for i, (lo, hi) in enumerate(bars):
                for edge, row in (("top", lo), ("bottom", hi)):
                    d = abs(y - row)
                    if d <= tol and (best is None or d < best[0]):
                        best = (d, source, i, edge,
                                lo if edge == "bottom" else hi)
        if best is not None:
            _, source, i, edge, other = best
            return ("edge", source, i, edge, other)
        # Interior hit: manual bars take precedence over auto suggestions.
        for source, bars in (("manual", self._manual), ("auto", self._auto)):
            for i, (lo, hi) in enumerate(bars):
                if lo + tol < y < hi - tol:
                    return ("inside", source, i)
        return None

    # ---------- interaction ---------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802
        # No zoom: the scan always fits the view.
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._pixmap_item is not None:
            pos = self._scene_pos(event)
            if 0 <= pos.y() <= self._image_size.height():
                hit = self._hit(pos.y())
                if hit and hit[0] == "edge":
                    _, source, index, edge, other = hit
                    if source == "manual":
                        self._selected = index
                        self._selected_auto = None
                        self._drag = (index, edge, other)
                        self._preview = (
                            index, int(round(min(other, pos.y()))),
                            int(round(max(other, pos.y()))))
                        self.barSelected.emit(index)
                    else:
                        self._selected = None
                        self._selected_auto = index
                        self._drag_auto = (index, edge, other)
                        self._auto_preview = (
                            index, int(round(min(other, pos.y()))),
                            int(round(max(other, pos.y()))))
                        self.autoSelected.emit(index)
                    self._redraw_overlays()
                elif hit and hit[0] == "inside":
                    _, source, index = hit
                    if source == "manual":
                        self._selected = index
                        self._selected_auto = None
                        self.barSelected.emit(index)
                    else:
                        self._selected = None
                        self._selected_auto = index
                        self.autoSelected.emit(index)
                    self._redraw_overlays()
                else:
                    self._selected = None
                    self._selected_auto = None
                    self._create_start = pos.y()
                    self._bar_temp = QGraphicsRectItem(
                        0, pos.y(), self._image_size.width(), 1)
                    self._bar_temp.setBrush(QBrush(QColor(51, 209, 122, 55)))
                    self._bar_temp.setPen(QPen(QColor(51, 209, 122, 230), 2))
                    self._bar_temp.setZValue(20)
                    self._scene.addItem(self._bar_temp)
                    self._redraw_overlays()
            event.accept()
            return
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = self._scene_pos(event)
        if 0 <= pos.x() <= self._image_size.width() and                 0 <= pos.y() <= self._image_size.height():
            self.cursorMoved.emit(pos)
        y = max(0.0, min(float(self._image_size.height()), pos.y()))
        if self._drag is not None:
            index, edge, other = self._drag
            # Keep the dragged edge on its own side of the bar: the top
            # cannot cross below the bottom, and the bottom cannot cross
            # above the top.
            if edge == "top":
                lo = min(y, other - 3.0)
                hi = other
            else:
                lo = other
                hi = max(y, other + 3.0)
            if hi - lo >= 3:
                self._preview = (index, int(round(lo)), int(round(hi)))
                self._redraw_overlays()
            self.viewport().setCursor(Qt.SizeVerCursor)
            event.accept()
            return
        if self._drag_auto is not None:
            index, edge, other = self._drag_auto
            if edge == "top":
                lo = min(y, other - 3.0)
                hi = other
            else:
                lo = other
                hi = max(y, other + 3.0)
            if hi - lo >= 3:
                self._auto_preview = (index, int(round(lo)), int(round(hi)))
                self._redraw_overlays()
            self.viewport().setCursor(Qt.SizeVerCursor)
            event.accept()
            return
        if self._create_start is not None and self._bar_temp is not None:
            lo, hi = sorted((self._create_start, y))
            self._bar_temp.setRect(QRectF(0, lo, self._image_size.width(),
                                          max(1.0, hi - lo)))
            event.accept()
            return
        # Cursor feedback for the edge handles.
        hit = self._hit(y)
        if hit and hit[0] == "edge":
            self.viewport().setCursor(Qt.SizeVerCursor)
        elif hit and hit[0] == "inside":
            self.viewport().setCursor(Qt.PointingHandCursor)
        else:
            self.viewport().setCursor(Qt.CrossCursor)
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._pixmap_item is not None:
            pos = self._scene_pos(event)
            hit = self._hit(pos.y())
            if hit and hit[1] == "auto":
                self.autoAdopted.emit(hit[2])
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_auto is not None:
            if self._auto_preview is not None:
                i, lo, hi = self._auto_preview
                self.autoAdjusted.emit(i, lo, hi)
            self._drag_auto = None
            self._auto_preview = None
            self._redraw_overlays()
            event.accept()
            return
        if self._drag is not None:
            if self._preview is not None:
                i, lo, hi = self._preview
                self.barAdjusted.emit(i, lo, hi)
            self._drag = None
            self._preview = None
            self._redraw_overlays()
            event.accept()
            return
        if self._create_start is not None and self._bar_temp is not None:
            rect = self._bar_temp.rect()
            if self._bar_temp.scene() is self._scene:
                self._scene.removeItem(self._bar_temp)
            self._bar_temp = None
            y0, y1 = int(round(rect.top())), int(round(rect.bottom()))
            self._create_start = None
            if abs(y1 - y0) >= 3:
                self.barDrawn.emit(y0, y1)
            self._redraw_overlays()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bar Annotator")
        self.resize(1400, 950)
        self.scan_root: Path | None = None
        self.roll = ""
        self.scans: list[ScanItem] = []
        self.visible_indices: list[int] = []
        self.current: int | None = None
        self.store = AnnotationStore()
        self._ocr_lib: OcrLibrary | None = None
        self._auto_cache: dict[str, tuple[list, tuple]] = {}
        self._plan_meta: dict[tuple[str, int], dict] = {}
        self.selected_bar: int | None = None
        self.selected_auto: int | None = None
        self._loading = False
        self._dirty = False

        self._autosave = QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(700)
        self._autosave.timeout.connect(self._autosave_now)

        self._make_actions()
        self._make_ui()
        self._update_title()

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
        self.act_open = self._act("&Open roll directory…", "Ctrl+O",
                                  self.open_directory)
        self.act_open_ann = self._act("Open &annotations…", "Ctrl+Shift+O",
                                      self.open_annotations)
        self.act_save = self._act("&Save", "Ctrl+S", self.save)
        self.act_quit = self._act("&Quit", "Ctrl+Q", self.close)
        self.act_prev = self._act("Previous scan", "Left", lambda: self.step(-1))
        self.act_next = self._act("Next scan", "Space", lambda: self.step(1))
        self.act_next_unmarked = self._act("Next unmarked", "N",
                                           self.next_unmarked)
        self.act_accept = self._act("&Accept auto bars", "A", self.accept_auto)
        self.act_clear = self._act("&Clear manual bars", "C", self.clear_bars)
        self.act_delete = self._act("Delete selected bar", "Delete",
                                    self.delete_selected)

    def _make_ui(self):
        self.view = BarView()
        self.view.barDrawn.connect(self._on_bar_drawn)
        self.view.barAdjusted.connect(self._on_bar_adjusted)
        self.view.autoAdjusted.connect(self._on_auto_adjusted)
        self.view.autoAdopted.connect(self._on_auto_adopted)
        self.view.barSelected.connect(self._on_bar_selected)
        self.view.autoSelected.connect(self._on_auto_selected)
        self.view.cursorMoved.connect(self._on_cursor)
        self.header = QLabel("Open a roll directory")
        self.header.setStyleSheet("QLabel { padding: 4px 8px; color: #ddd; "
                                  "background: #242424; }")
        self.header.setWordWrap(True)
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.header)
        lay.addWidget(self.view, 1)
        self.setCentralWidget(central)

        self.scan_list = QListWidget()
        self.scan_list.currentRowChanged.connect(self._on_row_changed)
        self.filter = QComboBox()
        for label, value in (("All scans", "all"), ("Unmarked", "unmarked"),
                             ("Marked", "marked")):
            self.filter.addItem(label, value)
        self.filter.currentIndexChanged.connect(lambda _=0: self._build_list(True))
        self.list_count = QLabel("")
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.addWidget(self.filter)
        ll.addWidget(self.scan_list, 1)
        ll.addWidget(self.list_count)
        self.left_dock = QDockWidget("Scans", self)
        self.left_dock.setObjectName("scans")
        self.left_dock.setWidget(left)
        self.left_dock.setMinimumWidth(230)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)

        self.bar_table = QTableWidget(0, 4)
        self.bar_table.setHorizontalHeaderLabels(["Source", "y0", "y1", "height"])
        self.bar_table.verticalHeader().setVisible(False)
        self.bar_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.bar_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.bar_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.bar_table.horizontalHeader().setStretchLastSection(True)
        self.bar_table.itemSelectionChanged.connect(self._on_table_selection)
        self.bar_table.cellDoubleClicked.connect(self._on_table_double_click)
        self.section_label = QLabel("Sections: —")
        self.section_label.setWordWrap(True)
        self.section_label.setStyleSheet("font-family: monospace;")

        buttons = QHBoxLayout()
        for text, slot in (("Auto-detect", self.auto_detect),
                           ("Accept auto (A)", self.accept_auto),
                           ("Clear (C)", self.clear_bars),
                           ("Delete selected", self.delete_selected),
                           ("Save (Ctrl+S)", self.save)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)
        rl.addWidget(QLabel("Blank bars (manual = solid, auto = dashed)"))
        rl.addWidget(self.bar_table, 1)
        rl.addWidget(self.section_label)
        rl.addLayout(buttons)
        rl.addWidget(QLabel("Drag on the scan to add a bar."))
        rl.addWidget(QLabel("Drag a selected bar's top/bottom edge to adjust it."))
        rl.addWidget(QLabel("Auto bars: drag an edge or double-click to convert."))
        rl.addWidget(QLabel("Sections are numbered S1, S2, S3 top-to-bottom."))
        self.right_dock = QDockWidget("Bars", self)
        self.right_dock.setObjectName("bars")
        self.right_dock.setWidget(right)
        self.right_dock.setMinimumWidth(360)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)

        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        for a in (self.act_open, self.act_save, None, self.act_prev,
                  self.act_next, self.act_next_unmarked, None,
                  self.act_accept, self.act_clear, self.act_delete):
            tb.addSeparator() if a is None else tb.addAction(a)

        sb = QStatusBar()
        hint = QLabel("Drag to add a bar; drag its top/bottom edge to adjust")
        self.cursor_label = QLabel("")
        sb.addWidget(hint)
        sb.addPermanentWidget(self.cursor_label)
        self.setStatusBar(sb)

    # ---------- opening -------------------------------------------------

    def open_directory(self):
        d = QFileDialog.getExistingDirectory(
            self, "Open a roll directory of scans",
            str(Path(__file__).resolve().parent / "20261028-newspaper-scans"))
        if d:
            self.load_directory(Path(d))

    def load_directory(self, directory: Path, annotations: Path | None = None):
        directory = Path(directory).resolve()
        scans = discover_scans(directory)
        if not scans:
            QMessageBox.warning(self, "No scans",
                                f"No indexed scans found in\\n{directory}")
            return
        self.scan_root = directory
        self.roll = directory.name
        self.scans = scans
        self._ocr_lib = OcrLibrary(directory)
        self._auto_cache = {}
        # Imported late so plan_v2 can import this module's detector without
        # creating an import cycle.
        from plan_v2 import load_plan_scan_meta
        self._plan_meta = load_plan_scan_meta(PLAN_ROOT, self.roll)
        if annotations is not None:
            self.store = AnnotationStore(annotations)
            self.store.load()
        elif self.store.path is None or self.store.scan_dir != str(directory):
            self.store = AnnotationStore(default_annotation_path(directory))
            self.store.load()
        self.store.scan_dir = str(directory)
        self._build_list(False)
        target = 0
        for i, s in enumerate(self.scans):
            if not self.store.get_bars(s.key(self.roll)):
                target = i
                break
        self._select_index(target)
        self._update_title()

    def open_annotations(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Open bar annotations",
            str(Path(__file__).resolve().parent / "plans" / "ocr_1028"
                / "annotations"),
            "Annotations (*.json);;All files (*)")
        if not f:
            return
        store = AnnotationStore(Path(f))
        try:
            store.load()
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not load", str(exc))
            return
        d = Path(store.scan_dir) if store.scan_dir else None
        if d and d.is_dir():
            self.load_directory(d, annotations=Path(f))
        else:
            self.store = store
            self._update_title()
            self._refresh()

    # ---------- scan list ----------------------------------------------

    def _build_list(self, keep_current: bool):
        selected = self.current if keep_current else None
        self.scan_list.blockSignals(True)
        self.scan_list.clear()
        self.visible_indices = []
        for i, s in enumerate(self.scans):
            bars = self.store.get_bars(s.key(self.roll))
            want = self.filter.currentData()
            if want == "unmarked" and bars:
                continue
            if want == "marked" and not bars:
                continue
            text = f"{i + 1:04d}  {s.prefix.split('_')[0]}  {s.index:04d}"
            if bars:
                text += f"   {len(bars)} bar{'s' if len(bars) != 1 else ''}"
            self.scan_list.addItem(text)
            item = self.scan_list.item(self.scan_list.count() - 1)
            item.setToolTip(s.key(self.roll))
            if bars:
                item.setForeground(QColor("#79d99a"))
            self.visible_indices.append(i)
        self.scan_list.blockSignals(False)
        self.list_count.setText(f"{len(self.visible_indices)} / {len(self.scans)} scans")
        if self.visible_indices:
            row = 0
            if selected is not None and selected in self.visible_indices:
                row = self.visible_indices.index(selected)
            self.scan_list.blockSignals(True)
            self.scan_list.setCurrentRow(row)
            self.scan_list.blockSignals(False)
            new_index = self.visible_indices[row]
            if self.current != new_index:
                self._on_row_changed(row)
        else:
            self.current = None
            self._refresh()

    def _select_index(self, index: int):
        if index not in self.visible_indices:
            if not self.visible_indices:
                return
            index = self.visible_indices[0]
        self.scan_list.setCurrentRow(self.visible_indices.index(index))

    def _on_row_changed(self, row: int):
        if 0 <= row < len(self.visible_indices):
            self.load_scan(self.visible_indices[row])

    def step(self, delta: int):
        if not self.visible_indices:
            return
        row = max(0, min(len(self.visible_indices) - 1,
                         self.scan_list.currentRow() + delta))
        self.scan_list.setCurrentRow(row)

    def next_unmarked(self):
        if not self.scans:
            return
        start = self.current if self.current is not None else 0
        for k in range(1, len(self.scans) + 1):
            i = (start + k) % len(self.scans)
            if not self.store.get_bars(self.scans[i].key(self.roll)):
                if self.filter.currentData() != "all":
                    self.filter.setCurrentIndex(0)
                self._select_index(i)
                return

    # ---------- scan loading -------------------------------------------

    def load_scan(self, index: int):
        if not (0 <= index < len(self.scans)):
            return
        self.current = index
        self.selected_bar = None
        self.selected_auto = None
        scan = self.scans[index]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok = self.view.load(scan.path)
        finally:
            QApplication.restoreOverrideCursor()
        self.header.setText(f"{scan.prefix} · scan {scan.index:04d} · {scan.path.name}")
        self.header.setToolTip(str(scan.path))
        if not ok:
            self.statusBar().showMessage("Could not load scan", 8000)
        self._refresh()

    def _auto_bars(self, scan: ScanItem):
        key = scan.key(self.roll)
        if key in self._auto_cache:
            return self._auto_cache[key]
        pm = self._plan_meta.get((scan.prefix, scan.index))
        plan_bars = pm.get("blank_rows") if pm else []
        ocr_scan = self._ocr_lib.load(scan.path) if self._ocr_lib else None
        bars, bbox = detect_auto_bars(scan.path, ocr_scan, plan_bars)
        if pm and pm.get("bbox"):
            bbox = pm["bbox"]
        if bbox is not None:
            bars = [b for b in bars if b[1] >= bbox[0] and b[0] <= bbox[1]]
        self._auto_cache[key] = (bars, bbox or (0, 0, 0, 0))
        return self._auto_cache[key]

    def _manual_bars(self, scan: ScanItem):
        return self.store.get_bars(scan.key(self.roll))

    def _refresh(self):
        if self.current is None:
            self.bar_table.setRowCount(0)
            self.section_label.setText("Sections: —")
            self.view.set_bars([], [], [])
            return
        scan = self.scans[self.current]
        auto, bbox = self._auto_bars(scan)
        manual = self._manual_bars(scan)
        top, bottom = bbox[0], bbox[1]
        sections = sections_for(manual, top, bottom) if manual else []
        if not sections and auto:
            # Preview what accepting auto bars would produce.
            sections = sections_for(auto, top, bottom)
        if self.selected_bar is not None and self.selected_bar >= len(manual):
            self.selected_bar = None
        if self.selected_auto is not None and self.selected_auto >= len(auto):
            self.selected_auto = None
        self.view.set_bars(manual, auto, sections, self.selected_bar,
                           self.selected_auto)
        rows = []
        for lo, hi in manual:
            rows.append(("manual", lo, hi))
        for lo, hi in auto:
            rows.append(("auto", lo, hi))
        self.bar_table.blockSignals(True)
        self.bar_table.setRowCount(len(rows))
        for i, (src, lo, hi) in enumerate(rows):
            for col, v in enumerate((src, str(lo), str(hi), str(hi - lo))):
                item = QTableWidgetItem(v)
                if src == "manual":
                    item.setForeground(QColor("#79d99a"))
                else:
                    item.setForeground(QColor("#d9a441"))
                self.bar_table.setItem(i, col, item)
        if self.selected_bar is not None and self.selected_bar < len(manual):
            self.bar_table.selectRow(self.selected_bar)
        elif self.selected_auto is not None and self.selected_auto < len(auto):
            self.bar_table.selectRow(len(manual) + self.selected_auto)
        else:
            self.bar_table.clearSelection()
        self.bar_table.blockSignals(False)
        if manual:
            sec_text = ", ".join(f"S{i+1}={lo}-{hi}"
                                 for i, (lo, hi) in enumerate(sections))
            self.section_label.setText(
                f"Sections ({len(sections)}): {sec_text}")
        elif auto:
            sec_text = ", ".join(f"S{i+1}={lo}-{hi}"
                                 for i, (lo, hi) in enumerate(sections))
            self.section_label.setText(
                f"No manual bars; auto preview ({len(sections)}): {sec_text}")
        else:
            self.section_label.setText("No bars detected; drag on the scan to add one.")

    # ---------- edits ---------------------------------------------------

    def _on_bar_drawn(self, y0: int, y1: int):
        if self.current is None:
            return
        scan = self.scans[self.current]
        key = scan.key(self.roll)
        self.store.add_bar(key, y0, y1)
        self.selected_auto = None
        bars = self.store.get_bars(key)
        self.selected_bar = next((i for i, (lo, hi) in enumerate(bars)
                                  if lo <= y0 <= hi), len(bars) - 1)
        self._mark_dirty()
        self._refresh()
        self._build_list(True)
        self.statusBar().showMessage(f"Added manual bar {y0}-{y1}", 5000)

    def _on_bar_adjusted(self, index: int, y0: int, y1: int):
        if self.current is None:
            return
        scan = self.scans[self.current]
        key = scan.key(self.roll)
        bars = self.store.get_bars(key)
        if not (0 <= index < len(bars)):
            return
        lo, hi = sorted((int(y0), int(y1)))
        if hi - lo < 3:
            return
        bars[index] = [lo, hi]
        self.store.bars[key] = bars
        self.selected_bar = index
        self.selected_auto = None
        self._mark_dirty()
        self._refresh()
        self._build_list(True)
        self.statusBar().showMessage(
            f"Adjusted bar {index + 1}: {lo}-{hi}", 3000)

    def _on_auto_adjusted(self, index: int, y0: int, y1: int):
        if self.current is None:
            return
        scan = self.scans[self.current]
        key = scan.key(self.roll)
        auto, bbox = self._auto_bars(scan)
        if not (0 <= index < len(auto)):
            return
        lo, hi = sorted((int(y0), int(y1)))
        if hi - lo < 3:
            return
        # This suggestion is now a manual bar; drop it from the auto list.
        auto = list(auto)
        auto.pop(index)
        self._auto_cache[key] = (auto, bbox)
        self.store.add_bar(key, lo, hi)
        bars = self.store.get_bars(key)
        self.selected_bar = next((i for i, (a, b) in enumerate(bars)
                                  if a <= lo <= b), len(bars) - 1)
        self.selected_auto = None
        self._mark_dirty()
        self._refresh()
        self._build_list(True)
        self.statusBar().showMessage(
            f"Auto bar converted and adjusted: {lo}-{hi}", 4000)

    def _on_auto_adopted(self, index: int):
        if self.current is None:
            return
        scan = self.scans[self.current]
        key = scan.key(self.roll)
        auto, bbox = self._auto_bars(scan)
        if not (0 <= index < len(auto)):
            return
        lo, hi = auto[index]
        auto = list(auto)
        auto.pop(index)
        self._auto_cache[key] = (auto, bbox)
        self.store.add_bar(key, lo, hi)
        bars = self.store.get_bars(key)
        self.selected_bar = next((i for i, (a, b) in enumerate(bars)
                                  if a <= lo <= b), len(bars) - 1)
        self.selected_auto = None
        self._mark_dirty()
        self._refresh()
        self._build_list(True)
        self.statusBar().showMessage(
            "Auto bar converted to manual; drag its edges to adjust", 5000)

    def _on_auto_selected(self, index: int):
        self.selected_auto = index
        self.selected_bar = None
        if self.current is None:
            return
        manual = self.store.get_bars(self.scans[self.current].key(self.roll))
        row = len(manual) + index
        if 0 <= row < self.bar_table.rowCount():
            self.bar_table.blockSignals(True)
            self.bar_table.selectRow(row)
            self.bar_table.blockSignals(False)
        self.view.set_selected(None, index)

    def _on_table_double_click(self, row: int, _col: int):
        if self.current is None:
            return
        manual = self.store.get_bars(self.scans[self.current].key(self.roll))
        if row >= len(manual):
            self._on_auto_adopted(row - len(manual))

    def _on_bar_selected(self, index: int):
        self.selected_bar = index
        self.selected_auto = None
        if 0 <= index < self.bar_table.rowCount():
            self.bar_table.blockSignals(True)
            self.bar_table.selectRow(index)
            self.bar_table.blockSignals(False)
        self.view.set_selected(index)

    def _on_table_selection(self):
        row = self.bar_table.currentRow()
        if self.current is None or row < 0:
            self.selected_bar = None
            self.view.set_selected(None)
            return
        manual = self.store.get_bars(self.scans[self.current].key(self.roll))
        if row < len(manual):
            self.selected_bar = row
            self.selected_auto = None
            self.view.set_selected(row)
        else:
            auto_idx = row - len(manual)
            auto, _ = self._auto_bars(self.scans[self.current])
            if 0 <= auto_idx < len(auto):
                self.selected_bar = None
                self.selected_auto = auto_idx
                self.view.set_selected(None, auto_idx)
            else:
                self.selected_bar = None
                self.selected_auto = None
                self.view.set_selected(None)

    def accept_auto(self):
        if self.current is None:
            return
        scan = self.scans[self.current]
        auto, bbox = self._auto_bars(scan)
        key = scan.key(self.roll)
        if not auto:
            self.statusBar().showMessage("No auto bars to accept", 4000)
            return
        self.store.bars[key] = [[int(a), int(b)] for a, b in auto]
        self._auto_cache[key] = ([], bbox)
        self.selected_bar = None
        self.selected_auto = None
        self._mark_dirty()
        self._refresh()
        self._build_list(True)
        self.statusBar().showMessage(f"Accepted {len(auto)} auto bar(s)", 5000)

    def clear_bars(self):
        if self.current is None:
            return
        scan = self.scans[self.current]
        self.store.bars.pop(scan.key(self.roll), None)
        self.selected_bar = None
        self.selected_auto = None
        self._mark_dirty()
        self._refresh()
        self._build_list(True)

    def delete_selected(self):
        if self.current is None:
            return
        scan = self.scans[self.current]
        key = scan.key(self.roll)
        if self.selected_auto is not None:
            auto, bbox = self._auto_bars(scan)
            if 0 <= self.selected_auto < len(auto):
                auto = list(auto)
                auto.pop(self.selected_auto)
                self._auto_cache[key] = (auto, bbox)
                self.selected_auto = None
                self._refresh()
                self.statusBar().showMessage("Auto bar suggestion dismissed", 3000)
            return
        if self.selected_bar is None:
            return
        manual = self.store.get_bars(key)
        row = self.selected_bar
        if 0 <= row < len(manual):
            self.store.remove_bar(key, row)
            self.selected_bar = None
            self._mark_dirty()
            self._refresh()
            self._build_list(True)

    def auto_detect(self):
        if self.current is None:
            return
        scan = self.scans[self.current]
        self._auto_cache.pop(scan.key(self.roll), None)
        self._refresh()
        self.statusBar().showMessage("Auto-detected bars", 4000)

    # ---------- persistence --------------------------------------------

    def _mark_dirty(self):
        self._dirty = True
        self._autosave.start()

    def _autosave_now(self):
        if self.store.path is not None:
            try:
                self.store.save()
                self._dirty = False
            except OSError as exc:
                self.statusBar().showMessage(f"Save failed: {exc}", 8000)

    def save(self):
        if self.store.path is None:
            f, _ = QFileDialog.getSaveFileName(
                self, "Save bar annotations",
                str(Path(__file__).resolve().parent / "plans" / "ocr_1028"
                    / "annotations" / f"{self.roll or 'bars'}.json"),
                "Annotations (*.json)")
            if not f:
                return
            self.store.path = Path(f)
        self.store.save()
        self._dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Saved {self.store.path}", 5000)

    def _update_title(self):
        name = self.store.path.name if self.store.path else "(unsaved)"
        self.setWindowTitle(f"Bar Annotator — {self.roll or '(no roll)'} — {name}")

    def closeEvent(self, event):  # noqa: N802
        if self.store.path is not None:
            try:
                self.store.save()
            except OSError:
                pass
        event.accept()

    # ---------- view ----------------------------------------------------

    def _on_cursor(self, point: QPointF):
        self.cursor_label.setText(f"x={point.x():.0f} y={point.y():.0f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", nargs="?", help="roll directory")
    ap.add_argument("--annotations", help="annotation JSON to open")
    ap.add_argument("--at", type=int, help="jump to this scan number")
    args = ap.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Bar Annotator")
    win = MainWindow()
    win.show()
    if args.annotations:
        store = AnnotationStore(Path(args.annotations))
        store.load()
        if store.scan_dir and Path(store.scan_dir).is_dir():
            win.load_directory(Path(store.scan_dir), annotations=Path(args.annotations))
        else:
            win.store = store
            win._update_title()
    if args.scan_dir:
        win.load_directory(Path(args.scan_dir), annotations=Path(args.annotations)
                           if args.annotations else None)
    if args.at is not None and win.scans:
        idx = next((i for i, s in enumerate(win.scans)
                    if s.index >= args.at), None)
        if idx is not None:
            win._select_index(idx)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
