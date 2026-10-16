#!/usr/bin/env python3
"""Scan + OCR side-by-side reader for QA of OCR output.

The window shows one scan at a time:

    +--------------------------+------------------------------+
    |  OCR text (reading order)|  scan image (slippy map)     |
    +--------------------------+------------------------------+

The OCR words are laid out as running text on the left, in newspaper reading
order: the page is post-processed into columns (from the OCR boxes' x-centres)
and the text is emitted one column at a time, starting with the leftmost, top
to bottom within each column, with a rule and a "Column N" label between
them.  The scan on the right can be dragged to pan and zoomed with the mouse
wheel, exactly like a slippy map (a mipmap pyramid keeps it fast at every zoom
level).  Hovering a word on *either* side highlights the word under the cursor
on both sides -- the OCR word gets an amber highlight and is scrolled into
view, and its bounding box is outlined on the scan (with a small label showing
the text).  Arrow keys move to the previous / next scan in the same folder.

OCR data is read from a JSON sidecar for each scan.  Two shapes are handled:

    [{"text": "the quick brown", "bbox": [[x0,y0],[x1,y1]], "confidence": .98}]
    [{"text": "word", "bbox": [[x,y],[x,y],[x,y],[x,y]], "confidence": .99}]

i.e. either line-level entries (the bbox spans the whole line; the words in
the text are distributed proportionally across it) or word-level entries.
The bbox may have two or more corner points; its axis-aligned extent is used.

Usage:
    python scan_ocr_reader.py [SCAN-IMAGE-OR-DIRECTORY] [--ocr-dir DIR ...]
    python scan_ocr_reader.py 20261007-newspaper-scans/P71-1921-CENLIB9102

Opening a directory loads every image in it (natural sort).  The OCR sidecar
for each scan is looked up next to the scan, in a sibling ``ocr_results``
folder, or -- falling back -- in a sibling checkout's
``ocr-studio/ocr_results[/<run>]/<folder>`` directory (the layout this repo's
OCR runs use).  ``--ocr-dir`` forces a specific location and may be repeated.

Controls:
    Ctrl+O            open a directory of scans
    Ctrl+Shift+O      choose the OCR results directory
    Left / Right      previous / next scan in the folder
    PageUp / PgDown   previous / next scan
    Home / End        first / last scan
    Ctrl+= / Ctrl+-   zoom the scan in / out
    Ctrl+0            fit the scan to the window
    B                 toggle showing every OCR box on the scan
    drag              pan the scan
    wheel             zoom the scan about the cursor
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
MIN_MIP = 256          # smallest side of the smallest mipmap level, px
CACHE_LIMIT = 3        # scan pyramids kept in memory
DOC_CACHE_LIMIT = 8    # parsed OCR documents kept in memory

# Text panel
TEXT_BG = QColor("#1b1b1b")
TEXT_FG = QColor("#e8e8e8")
TEXT_DIM = QColor("#8a8a8a")
TEXT_HL_BG = QColor("#ffd54f")   # hovered / synced word
TEXT_HL_FG = QColor("#1b1b1b")

# Scan panel
SCAN_BG = QColor("#0d0d0d")
BOX_PEN = QColor(255, 90, 70)
BOX_FILL = QColor(255, 90, 70, 55)
ALL_BOX_PEN = QColor(120, 170, 255, 90)
LABEL_BG = QColor(20, 20, 20, 235)
LABEL_FG = QColor("#ffd54f")


def natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def collect_images(directory: Path) -> list[Path]:
    return sorted(
        (p for p in directory.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=natural_key,
    )


# --------------------------------------------------------------------------- #
# OCR data model
# --------------------------------------------------------------------------- #

@dataclass
class OcrWord:
    """One highlightable token, in image pixel coordinates."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float
    line: int
    column: int = 0

    @property
    def rect(self) -> QRectF:
        return QRectF(self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0)


@dataclass
class OcrDocument:
    words: list[OcrWord]
    n_lines: int
    path: Path | None = None
    n_columns: int = 1


def _bbox_rect(bbox) -> tuple[float, float, float, float] | None:
    """Axis-aligned (x0, y0, x1, y1) for a bbox of 2+ points or a flat 4."""
    if not bbox:
        return None
    if (isinstance(bbox, (list, tuple)) and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)):
        x0, y0, x1, y1 = (float(v) for v in bbox)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    pts = []
    for pt in bbox:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            pts.append((float(pt[0]), float(pt[1])))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _iter_entries(data):
    """Yield (text, bbox, confidence) from the various OCR JSON shapes."""
    if isinstance(data, dict):
        for key in ("words", "lines", "results", "items", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        return
    for entry in data:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text", entry.get("word", entry.get("value", "")))
        if not isinstance(text, str):
            text = str(text)
        conf = entry.get("confidence", entry.get("conf", entry.get("score", 1.0)))
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 1.0
        yield text, entry.get("bbox"), conf


def _split_words(text: str, x0, y0, x1, y1, conf, line: int,
                 column: int = 0) -> list[OcrWord]:
    """Split an OCR line into tokens, spreading them across its bbox."""
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return []
    span = max(1, len(text))
    width = max(0.0, x1 - x0)
    out = []
    for m in matches:
        wx0 = x0 + width * (m.start() / span)
        wx1 = x0 + width * (m.end() / span)
        out.append(OcrWord(m.group(), wx0, y0, wx1, y1, conf, line, column))
    return out


def _detect_columns(entries, img_w: float) -> tuple[list[int], int]:
    """Assign each entry to a newspaper column; return (column_of, n_columns).

    OCR lines in a column share a similar horizontal centre, so a histogram
    of the entries' x-centres has one peak per column and a valley at each
    gutter.  We cut at the low-density valleys, then fold away the empty /
    near-empty page-margin columns.  Works whether the skew shifts the
    gutters or not, because it looks at where the text sits, not at a single
    global vertical projection.
    """
    n = len(entries)
    if n == 0 or img_w <= 0:
        return [0] * n, 1

    nb = max(16, int(round(img_w / 50.0)))
    hist = [0] * nb
    centers: list[float] = []
    for _, (x0, _y0, x1, _y1), _ in entries:
        c = (x0 + x1) / 2.0
        centers.append(c)
        b = min(nb - 1, max(0, int(c / img_w * nb)))
        hist[b] += 1

    # Light smoothing so one noisy bin does not create a false gutter.
    kernel = (1, 2, 3, 2, 1)
    ksum = sum(kernel)
    smooth = []
    for i in range(nb):
        s = 0
        for j, k in enumerate(kernel):
            idx = i + j - 2
            if 0 <= idx < nb:
                s += hist[idx] * k
        smooth.append(s / ksum)

    peak = max(smooth) or 1.0
    valley = 0.15 * peak
    bounds = [0.0]
    i = 0
    while i < nb:
        if smooth[i] < valley:
            j = i
            while j < nb and smooth[j] < valley:
                j += 1
            bounds.append((i + j) / 2.0 * img_w / nb)
            i = j
        else:
            i += 1
    bounds.append(float(img_w))

    ncols = len(bounds) - 1
    raw: list[int] = []
    counts = [0] * ncols
    for c in centers:
        ci = min(ncols - 1, max(0, bisect.bisect_right(bounds, c) - 1))
        raw.append(ci)
        counts[ci] += 1

    min_count = max(3, int(0.01 * n))
    keep = [i for i, c in enumerate(counts) if c >= min_count]
    if not keep:
        keep = [max(range(ncols), key=lambda i: counts[i])]
    col_centers = [(bounds[i] + bounds[i + 1]) / 2.0 for i in range(ncols)]
    col_of: list[int] = []
    for c, ci in zip(centers, raw):
        if ci in keep:
            col_of.append(keep.index(ci))
        else:
            col_of.append(min(range(len(keep)),
                              key=lambda k: abs(col_centers[keep[k]] - c)))
    return col_of, len(keep)


def _build_document(entries, path: Path | None) -> OcrDocument:
    """Group entries into columns, then lines, then split them into words.

    ``entries`` is a list of ``(text, (x0, y0, x1, y1), confidence)``.  The
    entries are assigned to newspaper columns and emitted column by column,
    starting with the leftmost, and top to bottom within each column.  Entries
    whose vertical centres are close are treated as one visual line; within a
    line they are ordered left to right.  This turns word-level OCR output into
    lines and leaves line-level output as lines, giving a flat reading order of
    highlightable tokens.
    """
    if not entries:
        return OcrDocument([], 0, path, 1)
    img_w = max(e[1][2] for e in entries)
    col_of, n_columns = _detect_columns(entries, img_w)

    heights = sorted(max(2.0, e[1][3] - e[1][1]) for e in entries)
    median_h = heights[len(heights) // 2]
    # A fixed, page-wide tolerance derived from the median box height.  Using
    # the median (not a growing per-line extent) stops a tall box from
    # chaining unrelated lines together; the reference center of each line is
    # the first entry assigned to it.
    tol = 0.6 * median_h
    gap_allow = 0.9 * median_h

    # Column-major reading order: leftmost column first, top to bottom.
    order = sorted(range(len(entries)),
                   key=lambda i: (col_of[i], entries[i][1][1], entries[i][1][0]))
    words: list[OcrWord] = []
    line_no = -1
    current: dict | None = None
    for i in order:
        text, (x0, y0, x1, y1), conf = entries[i]
        ci = col_of[i]
        cy = (y0 + y1) / 2.0
        if (current is not None and current["col"] == ci
                and abs(cy - current["ref_cy"]) <= tol
                and x0 <= current["x1"] + gap_allow
                and current["x0"] <= x1 + gap_allow):
            current["x0"] = min(current["x0"], x0)
            current["x1"] = max(current["x1"], x1)
            current["y0"] = min(current["y0"], y0)
            current["y1"] = max(current["y1"], y1)
            target = current["line"]
        else:
            line_no += 1
            current = {"col": ci, "ref_cy": cy, "x0": x0, "x1": x1,
                       "y0": y0, "y1": y1, "line": line_no}
            target = line_no
        words.extend(_split_words(text, x0, y0, x1, y1, conf, target, ci))
    return OcrDocument(words, line_no + 1, path, n_columns)


def load_ocr_document(path: Path | None) -> OcrDocument | None:
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    entries = []
    for text, bbox, conf in _iter_entries(data):
        rect = _bbox_rect(bbox)
        if rect is None or not text.strip():
            continue
        entries.append((text, rect, conf))
    if not entries:
        return OcrDocument([], 0, path)
    return _build_document(entries, Path(path))


def _candidate_ocr_roots(scan_dir: Path, extra: list[Path]) -> list[Path]:
    """Directories that may hold OCR sidecars for ``scan_dir``."""
    roots: list[Path] = []
    scan_dir = scan_dir.resolve()

    def add(p: Path) -> None:
        try:
            if p.is_dir():
                roots.append(p)
        except OSError:
            pass

    for e in extra:
        add(Path(e).resolve())
    if not extra:
        add(scan_dir)
        add(scan_dir / "ocr_results")
        add(scan_dir.parent / "ocr_results")
        for base in [scan_dir.parent, *scan_dir.parents]:
            sib = base / "ocr-studio"
            if not sib.is_dir():
                continue
            ocr = sib / "ocr_results"
            if ocr.is_dir():
                add(ocr)
                children = [c for c in sorted(ocr.iterdir()) if c.is_dir()]
                # Prefer the scan-oriented run over the older page runs.
                children.sort(key=lambda c: (0 if "scan" in c.name else 1,
                                              c.name.startswith("_"), c.name))
                for child in children:
                    add(child)
            # Older layout: ocr-studio/<batch>/<roll>[/ocr_results].
            batch = scan_dir.parent.name
            add(sib / batch / scan_dir.name)
            add(sib / batch / scan_dir.name / "ocr_results")
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        key = str(r.resolve())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def build_ocr_index(scan_dir: Path, images: list[Path],
                    extra: list[Path]) -> dict[str, Path]:
    """Map scan stem -> OCR JSON, picking the best covering OCR root."""
    best: dict[str, Path] = {}
    best_score = -1
    for root in _candidate_ocr_roots(scan_dir, extra):
        index: dict[str, Path] = {}
        for source in (root, root / scan_dir.name):
            if not source.is_dir():
                continue
            for j in source.glob("*.json"):
                index.setdefault(j.stem, j)
        score = sum(1 for im in images if im.stem in index)
        if score > best_score:
            best_score = score
            best = index
    return best


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #

def _display_image(img: QImage) -> QImage:
    """Convert to a format that paints quickly and uses little memory."""
    fmt = img.format()
    if fmt in (QImage.Format.Format_Grayscale8, QImage.Format.Format_Grayscale16):
        return img
    if fmt == QImage.Format.Format_Indexed8:
        table = img.colorTable()
        if all(((c & 0xFF) == ((c >> 8) & 0xFF) == ((c >> 16) & 0xFF))
               for c in table):
            return img.convertToFormat(QImage.Format.Format_Grayscale8)
        return img.convertToFormat(QImage.Format.Format_RGB32)
    return img.convertToFormat(QImage.Format.Format_RGB32)


def load_pyramid(path: Path):
    """Return ``([(level_scale, QPixmap), ...], width, height)`` or None.

    Levels are 1.0, 0.5, 0.25, ... down to a minimum side of ``MIN_MIP`` so
    that a zoomed-out view never has to rescale the full-resolution scan.
    """
    img = QImage(str(path))
    if img.isNull():
        return None
    img = _display_image(img)
    w, h = img.width(), img.height()
    levels = [(1.0, QPixmap.fromImage(img))]
    ls = 0.5
    while min(w, h) * ls >= MIN_MIP:
        scaled = img.scaled(max(1, int(w * ls)), max(1, int(h * ls)),
                            Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        levels.append((ls, QPixmap.fromImage(scaled)))
        ls *= 0.5
    return levels, w, h


# --------------------------------------------------------------------------- #
# Left: the OCR text
# --------------------------------------------------------------------------- #

class OcrTextView(QWidget):
    """Reading-order OCR text where every word is a hover target."""

    wordHovered = Signal(int)
    hoverCleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._font = QFont()
        self._font.setPointSizeF(11.0)
        self._words: list[OcrWord] = []
        self._word_rects: list[QRectF] = []
        self._rows: dict[int, list[int]] = {}
        self._column_marks: list[tuple[float, int]] = []
        self._line_h = 16.0
        self._margin = 16.0
        self._hover = -1
        self._highlight = -1
        self._scroll: QScrollArea | None = None
        self._content_h = 0
        self._last_width = -1
        self._empty_message = "No OCR loaded"

    # -- content --------------------------------------------------------- #

    def set_scroll_area(self, area: QScrollArea) -> None:
        self._scroll = area

    @property
    def words(self) -> list[OcrWord]:
        return self._words

    def set_document(self, doc: OcrDocument | None, message: str = "") -> None:
        self._words = list(doc.words) if doc else []
        self._hover = -1
        self._highlight = -1
        if doc is None:
            self._empty_message = message or "No OCR found for this scan"
        elif not doc.words:
            self._empty_message = message or "OCR file is empty"
        self._last_width = -1
        self.relayout()

    # -- layout ---------------------------------------------------------- #

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.width() != self._last_width:
            self.relayout()

    def relayout(self) -> None:
        self._last_width = self.width()
        fm = QFontMetricsF(self._font)
        margin = 16.0
        space = fm.horizontalAdvance(" ")
        line_h = fm.height() * 1.28
        self._line_h = line_h
        self._margin = margin
        avail = max(120.0, self.width() - 2 * margin)
        x = margin
        y = margin
        prev_line = -1
        prev_col = -1
        rects: list[QRectF] = []
        rows: dict[int, list[int]] = {}
        column_marks: list[tuple[float, int]] = []
        for i, w in enumerate(self._words):
            ww = fm.horizontalAdvance(w.text)
            if prev_col != -1 and w.column != prev_col:
                # Start a new newspaper column, with a blank line and a rule.
                x = margin
                y += line_h * 1.9
                column_marks.append((y - line_h * 1.0, w.column + 1))
            elif prev_line != -1 and w.line != prev_line:
                x = margin
                y += line_h
            elif x + ww > margin + avail and x > margin:
                x = margin
                y += line_h
            rects.append(QRectF(x, y, ww, fm.height()))
            rows.setdefault(int(y // line_h), []).append(i)
            x += ww + space
            prev_line = w.line
            prev_col = w.column
        self._word_rects = rects
        self._rows = rows
        self._column_marks = column_marks
        self._content_h = int(y + line_h + margin)
        self.setMinimumHeight(self._content_h)
        self.updateGeometry()
        self.update()

    # -- painting -------------------------------------------------------- #

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), TEXT_BG)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        if not self._words:
            painter.setFont(self._font)
            painter.setPen(TEXT_DIM)
            painter.drawText(self.rect().adjusted(20, 20, -20, -20),
                             Qt.AlignmentFlag.AlignHCenter |
                             Qt.AlignmentFlag.AlignTop,
                             self._empty_message)
            return

        painter.setFont(self._font)
        visible = QRectF(event.rect())

        for mark_y, col_no in self._column_marks:
            if mark_y < visible.top() - 40 or mark_y > visible.bottom() + 40:
                continue
            painter.setPen(QColor("#3d3d3d"))
            painter.drawLine(QPointF(self._margin, mark_y),
                             QPointF(self.width() - self._margin, mark_y))
            painter.setPen(QColor("#6f6f6f"))
            painter.drawText(QRectF(self._margin, mark_y + 1.0,
                                    self.width() - 2 * self._margin,
                                    self._line_h * 0.8),
                             int(Qt.AlignmentFlag.AlignLeft |
                                 Qt.AlignmentFlag.AlignTop),
                             f"Column {col_no}")

        for i, (w, r) in enumerate(zip(self._words, self._word_rects)):
            if not r.intersects(visible.adjusted(-80, -20, 80, 20)):
                continue
            active = i == self._highlight or i == self._hover
            if active:
                painter.fillRect(r.adjusted(-1, -1, 1, 1), TEXT_HL_BG)
                painter.setPen(TEXT_HL_FG)
            else:
                painter.setPen(TEXT_FG)
            painter.drawText(r, int(Qt.AlignmentFlag.AlignLeft |
                                    Qt.AlignmentFlag.AlignVCenter), w.text)

    # -- interaction ----------------------------------------------------- #

    def _word_at(self, pos: QPointF) -> int:
        if not self._word_rects:
            return -1
        row = int(pos.y() // self._line_h)
        candidates = []
        for r in (row - 1, row, row + 1):
            candidates.extend(self._rows.get(r, ()))
        for i in candidates:
            if self._word_rects[i].contains(pos):
                return i
        return -1

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        i = self._word_at(event.position())
        if i != self._hover:
            self._hover = i
            self.update()
            if i >= 0:
                self.wordHovered.emit(i)
            else:
                self.hoverCleared.emit()

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hover != -1:
            self._hover = -1
            self.update()
            self.hoverCleared.emit()
        super().leaveEvent(event)

    def set_highlight(self, index: int, ensure: bool = False) -> None:
        if index == self._highlight:
            if ensure and index >= 0:
                self._ensure_visible(index)
            return
        self._highlight = index
        self.update()
        if ensure:
            self._ensure_visible(index)

    def _ensure_visible(self, index: int) -> None:
        if self._scroll is None or not (0 <= index < len(self._word_rects)):
            return
        r = self._word_rects[index]
        self._scroll.ensureVisible(int(r.center().x()), int(r.center().y()), 40, 80)


# --------------------------------------------------------------------------- #
# Right: the scan as a slippy map
# --------------------------------------------------------------------------- #

class ScanView(QWidget):
    """Pannable, zoomable scan view with OCR word overlays."""

    wordHovered = Signal(int)
    hoverCleared = Signal()

    MIN_ZOOM = 0.05
    MAX_ZOOM = 32.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pyramid: list[tuple[float, QPixmap]] = []
        self._img_w = 0
        self._img_h = 0
        self._scale = 1.0
        self._fit_scale = 1.0
        self._origin = QPointF(0.0, 0.0)   # image coord at the widget's top-left
        self._auto_fit = True
        self._words: list[OcrWord] = []
        self._grid: dict[tuple[int, int], list[int]] = {}
        self._cell_w = 1.0
        self._cell_h = 1.0
        self._highlight = -1
        self._hover = -1
        self._show_boxes = False
        self._dragging = False
        self._drag_start = QPointF()
        self._drag_origin = QPointF()
        self._message = "Open a directory (Ctrl+O) or drop one here"

    # -- content --------------------------------------------------------- #

    def set_scan(self, loaded, words: list[OcrWord], message: str = "") -> None:
        self._words = list(words)
        self._highlight = -1
        self._hover = -1
        if loaded is None:
            self._pyramid = []
            self._img_w = self._img_h = 0
            self._grid = {}
            self._message = message or "Could not load scan"
            self.update()
            return
        self._pyramid, self._img_w, self._img_h = loaded
        self._build_grid()
        self._auto_fit = True
        self._fit()

    def _build_grid(self) -> None:
        """Bucket words into a coarse grid for O(1)-ish hover hit testing."""
        grid: dict[tuple[int, int], list[int]] = {}
        if self._words and self._img_w and self._img_h:
            cells = 48
            self._cell_w = max(1.0, self._img_w / cells)
            self._cell_h = max(1.0, self._img_h / cells)
            for i, w in enumerate(self._words):
                cx0 = max(0, int(w.x0 // self._cell_w))
                cx1 = max(0, int(w.x1 // self._cell_w))
                cy0 = max(0, int(w.y0 // self._cell_h))
                cy1 = max(0, int(w.y1 // self._cell_h))
                for cx in range(cx0, cx1 + 1):
                    for cy in range(cy0, cy1 + 1):
                        grid.setdefault((cx, cy), []).append(i)
        self._grid = grid

    def _fit(self) -> None:
        if not self._pyramid or self._img_w <= 0 or self._img_h <= 0:
            return
        vw = max(1, self.width())
        vh = max(1, self.height())
        self._fit_scale = min(vw / self._img_w, vh / self._img_h)
        self._scale = self._fit_scale
        self._origin = QPointF((self._img_w - vw / self._scale) / 2.0,
                               (self._img_h - vh / self._scale) / 2.0)
        self._auto_fit = True
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._auto_fit:
            self._fit()

    # -- coordinate mapping ---------------------------------------------- #

    def img_to_widget(self, x: float, y: float) -> QPointF:
        return QPointF((x - self._origin.x()) * self._scale,
                       (y - self._origin.y()) * self._scale)

    def widget_to_img(self, pos: QPointF) -> QPointF:
        return QPointF(pos.x() / self._scale + self._origin.x(),
                       pos.y() / self._scale + self._origin.y())

    def _pick_level(self, scale: float) -> tuple[float, QPixmap]:
        best = self._pyramid[0]
        for ls, px in self._pyramid:
            if ls >= scale:
                best = (ls, px)
            else:
                break
        return best

    # -- interaction ----------------------------------------------------- #

    def wheelEvent(self, event) -> None:  # noqa: N802
        if not self._pyramid:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 1.0 / 1.25
        pos = event.position()
        img_pt = self.widget_to_img(pos)
        new_scale = min(self.MAX_ZOOM, max(self.MIN_ZOOM, self._scale * factor))
        if new_scale == self._scale:
            return
        self._scale = new_scale
        self._origin = QPointF(img_pt.x() - pos.x() / self._scale,
                               img_pt.y() - pos.y() / self._scale)
        self._auto_fit = False
        self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._pyramid:
            self._dragging = True
            self._drag_start = event.position()
            self._drag_origin = QPointF(self._origin)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            delta = event.position() - self._drag_start
            self._origin = QPointF(self._drag_origin.x() - delta.x() / self._scale,
                                   self._drag_origin.y() - delta.y() / self._scale)
            self._auto_fit = False
            self.update()
            event.accept()
            return
        i = self._word_at(event.position())
        if i != self._hover:
            self._hover = i
            self.update()
            if i >= 0:
                self.wordHovered.emit(i)
            else:
                self.hoverCleared.emit()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            self._dragging = False
            self.unsetCursor()
            event.accept()

    def leaveEvent(self, event) -> None:  # noqa: N802
        if not self._dragging and self._hover != -1:
            self._hover = -1
            self.update()
            self.hoverCleared.emit()
        super().leaveEvent(event)

    def _word_at(self, pos: QPointF) -> int:
        if not self._words or not self._grid:
            return -1
        img = self.widget_to_img(pos)
        cell = (int(img.x() // self._cell_w), int(img.y() // self._cell_h))
        best = -1
        best_area = None
        for i in self._grid.get(cell, ()):
            w = self._words[i]
            if w.x0 <= img.x() <= w.x1 and w.y0 <= img.y() <= w.y1:
                area = (w.x1 - w.x0) * (w.y1 - w.y0)
                if best_area is None or area < best_area:
                    best = i
                    best_area = area
        return best

    def set_highlight(self, index: int, ensure: bool = False) -> None:
        if index == self._highlight:
            if ensure and index >= 0:
                self._ensure_visible(index)
            return
        self._highlight = index
        self.update()
        if ensure and index >= 0:
            self._ensure_visible(index)

    def _ensure_visible(self, index: int) -> None:
        if not (0 <= index < len(self._words)) or not self._pyramid:
            return
        w = self._words[index]
        tl = self.img_to_widget(w.x0, w.y0)
        br = self.img_to_widget(w.x1, w.y1)
        r = QRectF(tl, br).normalized()
        margin = 60.0
        dx = dy = 0.0
        if r.left() < margin:
            dx = margin - r.left()            # move the word right, into view
        elif r.right() > self.width() - margin:
            dx = (self.width() - margin) - r.right()   # move it left
        if r.top() < margin:
            dy = margin - r.top()             # move the word down, into view
        elif r.bottom() > self.height() - margin:
            dy = (self.height() - margin) - r.bottom()  # move it up
        if dx or dy:
            # widget = (image - origin) * scale, so shifting the widget rect
            # by (dx, dy) means origin moves by -(dx, dy) / scale.
            self._origin = QPointF(self._origin.x() - dx / self._scale,
                                   self._origin.y() - dy / self._scale)
            self._auto_fit = False
            self.update()

    def zoom_by(self, factor: float) -> None:
        if not self._pyramid:
            return
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        img_pt = self.widget_to_img(center)
        self._scale = min(self.MAX_ZOOM, max(self.MIN_ZOOM, self._scale * factor))
        self._origin = QPointF(img_pt.x() - center.x() / self._scale,
                               img_pt.y() - center.y() / self._scale)
        self._auto_fit = False
        self.update()

    def reset_zoom(self) -> None:
        self._fit()

    def toggle_boxes(self) -> None:
        self._show_boxes = not self._show_boxes
        self.update()

    # -- painting -------------------------------------------------------- #

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), SCAN_BG)
        if not self._pyramid:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        ls, pixmap = self._pick_level(self._scale)
        dest = QRectF(-self._origin.x() * self._scale,
                      -self._origin.y() * self._scale,
                      self._img_w * self._scale,
                      self._img_h * self._scale)
        painter.drawPixmap(dest, pixmap, QRectF(pixmap.rect()))

        if self._show_boxes:
            pen = painter.pen()
            pen.setColor(ALL_BOX_PEN)
            pen.setWidthF(1.0)
            painter.setPen(pen)
            for w in self._words:
                r = QRectF(self.img_to_widget(w.x0, w.y0), self.img_to_widget(w.x1, w.y1))
                if r.intersects(QRectF(self.rect())):
                    painter.drawRect(r)

        active = self._highlight if self._highlight >= 0 else self._hover
        if 0 <= active < len(self._words):
            self._draw_highlight(painter, active)

    def _draw_highlight(self, painter: QPainter, index: int) -> None:
        w = self._words[index]
        r = QRectF(self.img_to_widget(w.x0, w.y0), self.img_to_widget(w.x1, w.y1))
        r = r.normalized()
        painter.fillRect(r, BOX_FILL)
        pen = painter.pen()
        pen.setColor(BOX_PEN)
        pen.setWidthF(2.0)
        painter.setPen(pen)
        painter.drawRect(r)

        fm = QFontMetricsF(self.font())
        label = w.text
        tw = fm.horizontalAdvance(label) + 10
        th = fm.height() + 6
        lx = min(max(4.0, r.left()), max(4.0, self.width() - tw - 4.0))
        ly = r.top() - th - 4
        if ly < 4:
            ly = r.bottom() + 4
        box = QRectF(lx, ly, tw, th)
        painter.fillRect(box, LABEL_BG)
        painter.setPen(LABEL_FG)
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), label)


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #

class MainWindow(QMainWindow):
    def __init__(self, initial: Path | None = None,
                 ocr_dirs: list[Path] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Scan + OCR Reader")
        self.resize(1500, 950)

        self.images: list[Path] = []
        self.index = -1
        self.scan_dir: Path | None = None
        self.ocr_index: dict[str, Path] = {}
        self.ocr_dirs: list[Path] = [Path(p) for p in (ocr_dirs or [])]
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._docs: OrderedDict[str, OcrDocument | None] = OrderedDict()

        self.text_view = OcrTextView()
        self.scan_view = ScanView()

        self.text_scroll = QScrollArea()
        self.text_scroll.setWidgetResizable(True)
        self.text_scroll.setWidget(self.text_view)
        self.text_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_scroll.setStyleSheet("QScrollArea { border: none; }")
        self.text_view.set_scroll_area(self.text_scroll)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self.ocr_header = QLabel("No scan open")
        self.ocr_header.setStyleSheet(
            "QLabel { background: #141414; color: #b8b8b8; padding: 6px 10px; }")
        left_layout.addWidget(self.ocr_header)
        left_layout.addWidget(self.text_scroll, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.scan_view)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([560, 940])
        self.setCentralWidget(splitter)

        self.text_view.wordHovered.connect(self._on_text_hover)
        self.text_view.hoverCleared.connect(self._clear_hover)
        self.scan_view.wordHovered.connect(self._on_scan_hover)
        self.scan_view.hoverCleared.connect(self._clear_hover)

        self._make_actions()
        self._make_toolbar()
        self.statusBar().showMessage("Open a directory of scans (Ctrl+O)")

        self.setAcceptDrops(True)
        if initial is not None:
            self.load_path(Path(initial))

    # -- actions --------------------------------------------------------- #

    def _make_actions(self) -> None:
        self.act_open = QAction("Open folder…", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.open_directory)

        self.act_ocr = QAction("Choose OCR folder…", self)
        self.act_ocr.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.act_ocr.triggered.connect(self.choose_ocr_dir)

        self.act_prev = QAction("Previous scan", self)
        self.act_prev.setShortcuts([QKeySequence(Qt.Key.Key_Left),
                                    QKeySequence(Qt.Key.Key_PageUp)])
        self.act_prev.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_prev.triggered.connect(lambda: self.next_scan(-1))

        self.act_next = QAction("Next scan", self)
        self.act_next.setShortcuts([QKeySequence(Qt.Key.Key_Right),
                                    QKeySequence(Qt.Key.Key_PageDown)])
        self.act_next.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_next.triggered.connect(lambda: self.next_scan(1))

        self.act_first = QAction("First scan", self)
        self.act_first.setShortcut(QKeySequence(Qt.Key.Key_Home))
        self.act_first.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_first.triggered.connect(lambda: self.goto(0))

        self.act_last = QAction("Last scan", self)
        self.act_last.setShortcut(QKeySequence(Qt.Key.Key_End))
        self.act_last.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_last.triggered.connect(lambda: self.goto(len(self.images) - 1))

        self.act_zoom_in = QAction("Zoom in", self)
        self.act_zoom_in.setShortcut(QKeySequence("Ctrl+="))
        self.act_zoom_in.triggered.connect(lambda: self.scan_view.zoom_by(1.25))

        self.act_zoom_out = QAction("Zoom out", self)
        self.act_zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        self.act_zoom_out.triggered.connect(lambda: self.scan_view.zoom_by(0.8))

        self.act_zoom_fit = QAction("Fit", self)
        self.act_zoom_fit.setShortcut(QKeySequence("Ctrl+0"))
        self.act_zoom_fit.triggered.connect(self.scan_view.reset_zoom)

        self.act_boxes = QAction("Show all boxes", self)
        self.act_boxes.setCheckable(True)
        self.act_boxes.setShortcut(QKeySequence("B"))
        self.act_boxes.toggled.connect(lambda _: self.scan_view.toggle_boxes())

        for act in (self.act_prev, self.act_next, self.act_first, self.act_last,
                    self.act_zoom_in, self.act_zoom_out, self.act_zoom_fit,
                    self.act_boxes):
            self.addAction(act)

        menu = self.menuBar().addMenu("&File")
        menu.addAction(self.act_open)
        menu.addAction(self.act_ocr)
        view = self.menuBar().addMenu("&View")
        view.addAction(self.act_prev)
        view.addAction(self.act_next)
        view.addAction(self.act_first)
        view.addAction(self.act_last)
        view.addSeparator()
        view.addAction(self.act_zoom_in)
        view.addAction(self.act_zoom_out)
        view.addAction(self.act_zoom_fit)
        view.addSeparator()
        view.addAction(self.act_boxes)

    def _make_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        self.addToolBar(bar)
        bar.addAction(self.act_open)
        bar.addAction(self.act_ocr)
        bar.addSeparator()
        bar.addAction(self.act_prev)
        bar.addAction(self.act_next)
        bar.addAction(self.act_first)
        bar.addAction(self.act_last)
        bar.addSeparator()
        bar.addAction(self.act_zoom_in)
        bar.addAction(self.act_zoom_out)
        bar.addAction(self.act_zoom_fit)
        bar.addAction(self.act_boxes)

    # -- loading --------------------------------------------------------- #

    def open_directory(self) -> None:
        start = str(self.scan_dir or Path.cwd())
        chosen = QFileDialog.getExistingDirectory(self, "Open a folder of scans", start)
        if chosen:
            self.load_directory(Path(chosen))

    def choose_ocr_dir(self) -> None:
        start = str(self.ocr_dirs[0] if self.ocr_dirs else (self.scan_dir or Path.cwd()))
        chosen = QFileDialog.getExistingDirectory(self, "Choose OCR results folder", start)
        if not chosen:
            return
        self.ocr_dirs = [Path(chosen)]
        if self.scan_dir is not None:
            self.ocr_index = build_ocr_index(self.scan_dir, self.images, self.ocr_dirs)
            self.goto(self.index)

    def load_path(self, path: Path) -> None:
        if path.is_dir():
            self.load_directory(path)
        elif path.is_file():
            self.load_directory(path.parent, select=path)

    def load_directory(self, directory: Path, select: Path | None = None) -> None:
        images = collect_images(directory)
        if not images:
            self.statusBar().showMessage(f"No images in {directory}")
            return
        self.scan_dir = directory
        self.images = images
        self.ocr_index = build_ocr_index(directory, images, self.ocr_dirs)
        index = 0
        if select is not None:
            try:
                index = images.index(select)
            except ValueError:
                index = 0
        self.goto(index)

    def goto(self, index: int) -> None:
        if not self.images:
            return
        self.index = max(0, min(len(self.images) - 1, index))
        path = self.images[self.index]

        ocr_path = self.ocr_index.get(path.stem)
        doc = None
        if ocr_path is not None:
            key = str(ocr_path)
            if key in self._docs:
                doc = self._docs[key]
                self._docs.move_to_end(key)
            else:
                doc = load_ocr_document(ocr_path)
                self._docs[key] = doc
                while len(self._docs) > DOC_CACHE_LIMIT:
                    self._docs.popitem(last=False)
        if doc is None:
            message = (f"No OCR JSON found for {path.name}\n"
                       f"(looked in {len(self.ocr_index)} known sidecars; "
                       f"use Ctrl+Shift+O to choose a folder)")
            self.text_view.set_document(None, message)
        else:
            self.text_view.set_document(doc)
        words = doc.words if doc is not None else []

        loaded = self._cache.get(str(path))
        if loaded is None:
            loaded = load_pyramid(path)
            if loaded is not None:
                self._cache[str(path)] = loaded
                self._cache.move_to_end(str(path))
                while len(self._cache) > CACHE_LIMIT:
                    self._cache.popitem(last=False)
        self.scan_view.set_scan(loaded, words)

        self._update_titles(path, doc)

    def _update_titles(self, path: Path, doc: OcrDocument | None) -> None:
        n = len(self.images)
        folder = self.scan_dir.name if self.scan_dir else ""
        self.setWindowTitle(f"{folder} — {path.name} — Scan + OCR Reader")
        if doc is None:
            ocr_note = "no OCR"
        else:
            ocr_note = (f"{len(doc.words)} words / {doc.n_lines} lines / "
                        f"{doc.n_columns} columns")
        self.ocr_header.setText(f"{path.name}   ·   {ocr_note}")
        self.statusBar().showMessage(
            f"{self.index + 1} / {n}   ·   {path.name}   ·   {ocr_note}   ·   "
            f"OCR folder: {self._ocr_root_label()}")

    def _ocr_root_label(self) -> str:
        if not self.ocr_index:
            return "not found"
        try:
            return str(next(iter(self.ocr_index.values())).parent)
        except StopIteration:
            return "not found"

    def next_scan(self, delta: int) -> None:
        if self.images:
            self.goto(self.index + delta)

    # -- hover sync ------------------------------------------------------ #

    def _on_text_hover(self, index: int) -> None:
        self.scan_view.set_highlight(index, ensure=True)
        self._describe(index)

    def _on_scan_hover(self, index: int) -> None:
        self.text_view.set_highlight(index, ensure=True)
        self.scan_view.set_highlight(index)
        self._describe(index)

    def _clear_hover(self) -> None:
        self.text_view.set_highlight(-1)
        self.scan_view.set_highlight(-1)
        if self.index >= 0:
            self.statusBar().showMessage(
                f"{self.index + 1} / {len(self.images)}   ·   "
                f"{self.images[self.index].name}")

    def _describe(self, index: int) -> None:
        words = self.text_view.words
        if not (0 <= index < len(words)):
            return
        word = words[index]
        if self.index < 0:
            return
        self.statusBar().showMessage(
            f"{self.index + 1} / {len(self.images)}   ·   "
            f"{self.images[self.index].name}   ·   "
            f"word {index + 1}: “{word.text}”   (conf {word.conf:.3f})")

    # -- drag & drop ----------------------------------------------------- #

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                self.load_directory(path)
                return
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                self.load_directory(path.parent, select=path)
                return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?",
                        help="scan image or directory of scans to open")
    parser.add_argument("--ocr-dir", action="append", default=[], metavar="DIR",
                        help="folder holding the OCR JSON sidecars (repeatable)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Scan + OCR Reader")
    app.setStyleSheet("""
        QMainWindow, QWidget { background: #1b1b1b; color: #e8e8e8; }
        QScrollArea { border: none; }
        QStatusBar { background: #141414; color: #b0b0b0; }
        QToolBar { background: #212121; border: none; spacing: 4px; }
        QSplitter::handle { background: #000000; }
    """)
    window = MainWindow(Path(args.path) if args.path else None,
                         [Path(d) for d in args.ocr_dir])
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
