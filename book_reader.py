#!/usr/bin/env python3
"""Book-style two-page image viewer, like archive.org's BookReader.

Pages are shown two at a time as spreads; the first (cover) page and the
last page are shown alone, like a real book.

Projects are stored as JSON and hold the ordered list of page directories,
the pages to skip (duplicates), and the confirmed page pairs:

    {
        "version": 1,
        "directories": ["pages/P81-4720", "pages/P81-4721"],
        "skip": ["page005-scans011-011.png"],
        "confirmed_pairs": [["a.png", "b.png"], ...],
        "flagged_spreads": [["a.png", "b.png"], ...],
        "blank_pages": ["blank:page009-...png"],   # blank inserted before that page
        "reverse": false                            # reverse (from the back) mode
    }

Usage:
    python book_reader.py [directory-or-project.json]

Controls:
    Ctrl+O            open a directory of page images
    Ctrl+N            new (empty) project
    Ctrl+Shift+O      open a project file
    Ctrl+S / Shift+Ctrl+S  save project / save as
    Ctrl+E            edit project (directories, skip list)
    C                 confirm the pair, then jump to the next unconfirmed pair
    U                 unconfirm the showing pair (toggle off only)
    Z / X             skip the left / right shown page (toggle)
    N                 jump to the next unconfirmed pair
    F                 flag the spread for review (toggle)
    G                 jump to the next flagged spread
    I / Shift+I       insert a blank page before / after the spread (toggle)
    Right / Space     next spread (Left in reverse mode)
    Left              previous spread (Right in reverse mode)
    PageDown / Up     next / previous spread
    Home / End        first / last spread
    Ctrl+= / Ctrl+-   zoom in / out
    Ctrl+0            reset zoom
    Ctrl+G            go to page...
    W / Shift+W       fit page / fit width
    R                 toggle reverse (read from the back, last page first)
    Ctrl+wheel        zoom
    drag              pan when zoomed in
"""

from __future__ import annotations

import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QImage, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QStatusBar,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

GAP = 12          # px between the two pages of a spread
MARGIN = 16       # px padding around the spread
CAPTION_H = 22    # px strip under each page showing its number

BASE_H = 1000.0   # normalized page height used for layout math
BLANK_ASPECT = 0.7  # width/height used for inserted blank pages


def natural_key(path: Path):
    """Sort key that orders page002 before page010."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def collect_images(directory: Path) -> list[Path]:
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=natural_key,
    )


def build_spreads(n: int) -> list[tuple[int, ...]]:
    """Index 0 (the cover) alone, then pages in pairs, last page alone if odd."""
    if n == 0:
        return []
    spreads = [(0,)]
    i = 1
    while i < n:
        if i + 1 < n:
            spreads.append((i, i + 1))
            i += 2
        else:
            spreads.append((i,))
            i += 1
    return spreads


def build_spreads_from_back(n: int) -> list[tuple[int, ...]]:
    """Mirror image of build_spreads: the last page alone, then pairs working
    backwards, ending with the cover alone if the count is odd."""
    if n == 0:
        return []
    spreads = [(n - 1,)]
    i = n - 2
    while i >= 0:
        if i - 1 >= 0:
            spreads.append((i - 1, i))
            i -= 2
        else:
            spreads.append((i,))
            i -= 1
    return spreads


class Project:
    """A book project: ordered page directories, pages to skip, confirmed pairs.

    JSON format (paths relative to the project file when possible):

        {
            "version": 1,
            "directories": ["pages/P81-4720", "pages/P81-4721"],
            "skip": ["page005-scans011-011.png"],
            "confirmed_pairs": [["a.png", "b.png"], ...]
        }
    """

    VERSION = 1

    def __init__(self) -> None:
        self.path: Path | None = None
        self.directories: list[Path] = []
        self.skip: set[str] = set()
        self.confirmed: set[frozenset[str]] = set()
        self.flagged: set[frozenset[str]] = set()   # spreads marked for review
        self.blanks: set[str] = set()   # "blank:<pagekey>" = blank page inserted before that page
        self.reverse: bool = False                   # reverse (read from the back) mode

    # ----- page identity ---------------------------------------------

    def _base(self) -> Path | None:
        return self.path.parent.resolve() if self.path else None

    def key_for(self, page_dir: Path, page: Path) -> str:
        """Stable identifier for a page, relative to the project file if possible."""
        base = self._base()
        page = page.resolve()
        if base is not None:
            try:
                return str(page.relative_to(base))
            except ValueError:
                pass
        return str(page)

    # ----- serialization ----------------------------------------------

    def to_dict(self) -> dict:
        base = self._base()

        def rel(p: Path) -> str:
            if base is not None:
                try:
                    return str(p.resolve().relative_to(base))
                except ValueError:
                    pass
            return str(p.resolve())

        return {
            "version": self.VERSION,
            "directories": [rel(d) for d in self.directories],
            "skip": sorted(self.skip),
            "confirmed_pairs": sorted([sorted(pair) for pair in self.confirmed]),
            "flagged_spreads": sorted([sorted(f) for f in self.flagged]),
            "blank_pages": sorted(self.blanks),
            "reverse": self.reverse,
        }

    def update_from_dict(self, data: dict, project_dir: Path) -> None:
        self.directories = [
            Path(p) if Path(p).is_absolute() else project_dir / p
            for p in data.get("directories", [])
        ]
        self.skip = set(data.get("skip", []))
        self.confirmed = {frozenset(pair) for pair in data.get("confirmed_pairs", [])}
        self.flagged = {frozenset(f) for f in data.get("flagged_spreads", [])}
        self.blanks = set(data.get("blank_pages", []))
        self.reverse = bool(data.get("reverse", False))

    def load(self, path: Path) -> None:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.update_from_dict(data, path.resolve().parent)
        self.path = path.resolve()

    def save(self, path: Path | None = None) -> None:
        if path is not None:
            self.path = Path(path).resolve()
        if self.path is None:
            raise ValueError("no project file path")
        # Store keys relative to the new location if possible.
        old_base = self.path.parent
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def title(self) -> str:
        return self.path.name if self.path else "(unsaved project)"


class ProjectDialog(QDialog):
    """Editor for the project: page directories and the skip list."""

    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project")
        self.resize(520, 480)
        self._dirs = [str(d) for d in project.directories]
        self._skip = sorted(project.skip)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Page directories (in reading order):"))
        self.dir_list = QListWidget(self)
        self.dir_list.addItems(self._dirs)
        layout.addWidget(self.dir_list)
        dir_buttons = QHBoxLayout()
        for text, slot in (
            ("Add…", self._add_dir),
            ("Remove", self._remove_dir),
            ("Up", lambda: self._move_dir(-1)),
            ("Down", lambda: self._move_dir(1)),
        ):
            b = QPushButton(text, self)
            b.clicked.connect(slot)
            dir_buttons.addWidget(b)
        dir_buttons.addStretch(1)
        layout.addLayout(dir_buttons)

        layout.addWidget(QLabel("Skipped pages (duplicates):"))
        self.skip_list = QListWidget(self)
        self.skip_list.addItems(self._skip)
        layout.addWidget(self.skip_list)
        skip_buttons = QHBoxLayout()
        for text, slot in (
            ("Remove selected", self._remove_skip),
            ("Clear", lambda: self.skip_list.clear()),
        ):
            b = QPushButton(text, self)
            b.clicked.connect(slot)
            skip_buttons.addWidget(b)
        skip_buttons.addStretch(1)
        layout.addLayout(skip_buttons)

        ok_cancel = QHBoxLayout()
        ok = QPushButton("OK", self)
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel", self)
        cancel.clicked.connect(self.reject)
        ok_cancel.addStretch(1)
        ok_cancel.addWidget(ok)
        ok_cancel.addWidget(cancel)
        layout.addLayout(ok_cancel)

    # ----- directory list handlers -------------------------------------

    def _add_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Add page directory")
        if d:
            self.dir_list.addItem(d)

    def _remove_dir(self) -> None:
        for item in self.dir_list.selectedItems():
            self.dir_list.takeItem(self.dir_list.row(item))

    def _move_dir(self, delta: int) -> None:
        row = self.dir_list.currentRow()
        new_row = row + delta
        if not (0 <= row < self.dir_list.count()) or not (0 <= new_row < self.dir_list.count()):
            return
        item = self.dir_list.takeItem(row)
        self.dir_list.insertItem(new_row, item)
        self.dir_list.setCurrentRow(new_row)

    def _remove_skip(self) -> None:
        for item in self.skip_list.selectedItems():
            self.skip_list.takeItem(self.skip_list.row(item))

    # ----- results -------------------------------------------------------

    def directories(self) -> list[str]:
        return [self.dir_list.item(i).text() for i in range(self.dir_list.count())]

    def skip(self) -> list[str]:
        return [self.skip_list.item(i).text() for i in range(self.skip_list.count())]


class PageCache(QObject):
    """LRU cache of full-resolution QImages with background prefetching.

    Emits ``imageLoaded`` (from a worker thread, auto-queued to the GUI
    thread for QObject receivers) whenever a prefetch finishes.
    """

    LIMIT = 32
    imageLoaded = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._cache: OrderedDict[str, QImage] = OrderedDict()
        self._missing: set[str] = set()

    def get_sync(self, path: str | None) -> QImage | None:
        """Return the image, loading it synchronously if needed."""
        if path is None:
            return None
        img = self._hit(path)
        if img is not None:
            return img
        img = QImage(path)
        if img.isNull():
            self._missing.add(path)
            return None
        self._insert(path, img)
        return img

    def prefetch(self, paths) -> None:
        for path in paths:
            if path is None or self._hit(path) is not None or path in self._missing:
                continue
            QThreadPool.globalInstance().start(_LoadTask(self, path))

    def _hit(self, path: str) -> QImage | None:
        img = self._cache.get(path)
        if img is not None:
            self._cache.move_to_end(path)
        return img

    def _insert(self, path: str, img: QImage) -> None:
        self._cache[path] = img
        self._cache.move_to_end(path)
        while len(self._cache) > self.LIMIT:
            self._cache.popitem(last=False)


class _LoadTask(QRunnable):
    def __init__(self, cache: PageCache, path: str) -> None:
        super().__init__()
        self.cache = cache
        self.path = path

    def run(self) -> None:
        if self.cache._hit(self.path) is not None:
            return
        img = QImage(self.path)
        if img.isNull():
            self.cache._missing.add(self.path)
            return
        self.cache._insert(self.path, img)
        try:
            self.cache.imageLoaded.emit(self.path)
        except RuntimeError:
            pass  # cache (or app) already torn down


class SpreadWidget(QWidget):
    """Draws the current spread (one or two pages) scaled to fit the viewport."""

    zoomRequested = Signal(float)   # multiplicative zoom step, e.g. 1.25 / 0.8

    def __init__(self, cache: PageCache, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cache = cache
        cache.imageLoaded.connect(self._on_image_loaded)
        self._pages: list[Path] = []
        self._keys: list[str] = []
        self._spreads: list[tuple[int, ...]] = []
        self._spreads_back: list[tuple[int, ...]] = []
        self._confirmed: set[frozenset[str]] = set()
        self._flagged: set[frozenset[str]] = set()
        self._index = 0
        self._rtl = False
        self._fit_mode = "page"     # "page" or "width"
        self._zoom = 1.0
        self._viewport = QSize(800, 600)
        self._items: list[tuple[QRectF, QPixmap | None, str, bool]] = []
        self._pan_origin: QPointF | None = None
        self._pan_scroll_start = QPointF(0, 0)
        self._scroll_area: QScrollArea | None = None
        self.setAutoFillBackground(True)

    # ---------- book --------------------------------------------------

    def set_book(self, pages: list[Path], keys: list[str] | None = None) -> None:
        self._pages = pages
        self._keys = list(keys) if keys is not None else [str(p) for p in pages]
        self._spreads = build_spreads(len(pages))
        self._spreads_back = build_spreads_from_back(len(pages))
        self._index = 0
        self.relayout()

    def set_confirmed(self, confirmed: set[frozenset[str]]) -> None:
        self._confirmed = set(confirmed)
        self.relayout()

    def set_flagged(self, flagged: set[frozenset[str]]) -> None:
        self._flagged = set(flagged)
        self.relayout()

    def spread_count(self) -> int:
        return len(self._spreads)

    def spread_index(self) -> int:
        return self._index

    def _spread_at(self, display_index: int) -> tuple[int, ...] | None:
        """Map a display position to its spread; reversed when in reverse mode."""
        spreads = self._spreads_back if self._rtl else self._spreads
        if not spreads or not (0 <= display_index < len(spreads)):
            return None
        return spreads[display_index]

    def current_page_numbers(self) -> tuple[int, ...]:
        spread = self._spread_at(self._index)
        if spread is None:
            return ()
        return tuple(i + 1 for i in spread)

    def current_page_keys(self) -> tuple[str, ...]:
        spread = self._spread_at(self._index)
        if spread is None:
            return ()
        return tuple(self._keys[i] for i in spread)

    def visible_page_keys(self) -> tuple[str, ...]:
        """Keys in on-screen order: left page first, right page second."""
        return tuple(self._keys[i] for i in self._visible_entries())

    def is_current_pair_confirmed(self) -> bool:
        keys = self.current_page_keys()
        return len(keys) == 2 and frozenset(keys) in self._confirmed

    def is_current_spread_flagged(self) -> bool:
        keys = self.current_page_keys()
        return bool(keys) and frozenset(keys) in self._flagged

    def set_spread_index(self, index: int) -> None:
        if not self._spreads:
            return
        index = max(0, min(index, len(self._spreads) - 1))
        self._index = index
        self.relayout()
        self.prefetch_neighbors()
    def next_spread(self, delta: int) -> None:
        self.set_spread_index(self._index + delta)

    # ---------- options -----------------------------------------------

    def set_rtl(self, rtl: bool) -> None:
        self._rtl = rtl
        self.relayout()

    def set_fit_mode(self, mode: str) -> None:
        self._fit_mode = mode
        self.relayout()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.2, min(8.0, zoom))
        self.relayout()

    def zoom(self) -> float:
        return self._zoom

    def set_viewport_size(self, size: QSize) -> None:
        if size != self._viewport:
            self._viewport = size
            self.relayout()

    def setScrollArea(self, area: QScrollArea) -> None:  # noqa: N802
        self._scroll_area = area

    # ---------- layout --------------------------------------------------

    def _visible_entries(self) -> list[int]:
        spread = self._spread_at(self._index)
        if spread is None:
            return []
        entries = list(spread)
        if self._rtl:
            entries.reverse()   # mirror the pair for right-to-left reading
        return entries

    def relayout(self) -> None:
        self._items = []
        entries = self._visible_entries()
        if not entries:
            self.setFixedSize(max(self._viewport.width(), 200), max(self._viewport.height(), 200))
            self.update()
            return

        images = []
        for i in entries:
            page = self._pages[i]
            images.append(None if page is None else self._cache.get_sync(str(page)))
        aspects = []
        for i, img in enumerate(images):
            if img is not None and img.height():
                aspects.append(img.width() / img.height())
            elif self._pages[entries[i]] is None:
                aspects.append(BLANK_ASPECT)   # inserted blank page
            else:
                aspects.append(0.65)           # placeholder while loading
        widths = [a * BASE_H for a in aspects]

        if len(entries) == 1:
            content_w = widths[0] * 2 + GAP     # lone page: blank half beside it
        else:
            content_w = sum(widths) + GAP

        avail_w = max(50, self._viewport.width() - 2 * MARGIN)
        avail_h = max(50, self._viewport.height() - 2 * MARGIN - CAPTION_H)
        if self._fit_mode == "width":
            scale = avail_w / content_w
        else:
            scale = min(avail_w / content_w, avail_h / BASE_H)
        scale *= self._zoom

        page_h = BASE_H * scale
        w_px = int(content_w * scale) + 2 * MARGIN
        h_px = int(page_h) + CAPTION_H + 2 * MARGIN
        self.setFixedSize(max(w_px, 200), max(h_px, 200))

        # Horizontal placement of each page.
        if len(entries) == 1:
            xs = [MARGIN + widths[0] * scale + GAP]   # lone page on the right half
        else:
            xs = []
            x = MARGIN
            for w in widths:
                xs.append(x)
                x += w * scale + GAP

        n_pages = len(self._pages)
        pair_confirmed = (
            len(entries) == 2
            and frozenset(self._keys[i] for i in entries) in self._confirmed
        )
        flagged = frozenset(self._keys[i] for i in entries) in self._flagged
        suffix = ("  ✓" if pair_confirmed else "") + ("  ⚑" if flagged else "")
        for i, page_idx in enumerate(entries):
            rect = QRectF(xs[i], MARGIN, widths[i] * scale, page_h)
            pixmap = None
            blank = self._pages[page_idx] is None
            img = images[i]
            if not blank and img is not None:
                target = QSize(int(rect.width()), int(rect.height()))
                pixmap = QPixmap.fromImage(img.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self._items.append((rect, pixmap, f"{page_idx + 1} / {n_pages}{suffix}", blank))

        self.update()

    def _on_image_loaded(self, path: str) -> None:
        # A prefetched neighbor arrived; repaint only if it is visible now.
        entries = self._visible_entries()
        if any(
            self._pages[i] is not None and str(self._pages[i]) == path
            for i in entries
        ):
            self.relayout()

    def prefetch_neighbors(self) -> None:
        if not self._spreads:
            return
        paths: list[str] = []
        for idx in (self._index - 1, self._index + 1):
            spread = self._spread_at(idx)
            if spread is not None:
                paths.extend(
                    str(self._pages[i]) for i in spread if self._pages[i] is not None
                )
        self._cache.prefetch(paths)

    # ---------- painting --------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))

        if not self._items:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a directory (Ctrl+O) or drop one here")
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        for rect, pixmap, caption, blank in self._items:
            if blank:
                painter.fillRect(rect, QColor("#f7f5ee"))
                painter.setPen(QColor("#c9c5b8"))
                painter.drawText(rect, Qt.AlignCenter, "blank")
            elif pixmap is None:
                painter.fillRect(rect, QColor("#333333"))
                painter.setPen(QColor("#666666"))
                painter.drawText(rect, Qt.AlignCenter, "…")
            else:
                painter.drawPixmap(int(rect.left()), int(rect.top()), pixmap)
            font = painter.font()
            font.setPointSizeF(10)
            painter.setFont(font)
            painter.setPen(QColor("#9a9a9a"))
            cap_rect = QRectF(rect.left(), rect.bottom() + 4, rect.width(), CAPTION_H)
            painter.drawText(cap_rect, Qt.AlignHCenter | Qt.AlignTop, caption)
            painter.setPen(QColor("#3a3a3a"))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))

    # ---------- interaction -------------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:
            step = 1.25 if event.angleDelta().y() > 0 else 0.8
            self.zoomRequested.emit(step)
            event.accept()
        else:
            event.ignore()  # let the QScrollArea scroll

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._scroll_area is not None:
            self._pan_origin = event.position()
            self._pan_scroll_start = QPointF(
                self._scroll_area.horizontalScrollBar().value(),
                self._scroll_area.verticalScrollBar().value(),
            )
            self.setCursor(Qt.ClosedCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pan_origin is not None and self._scroll_area is not None:
            delta = event.position() - self._pan_origin
            self._scroll_area.horizontalScrollBar().setValue(int(self._pan_scroll_start.x() - delta.x()))
            self._scroll_area.verticalScrollBar().setValue(int(self._pan_scroll_start.y() - delta.y()))
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._pan_origin = None
        self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)


class BookScrollArea(QScrollArea):
    """QScrollArea that keeps the spread informed of the viewport size."""

    def __init__(self, spread: SpreadWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spread = spread
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignCenter)
        self.viewport().setStyleSheet("background: #141414;")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._spread.set_viewport_size(self.viewport().size())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Book Reader")
        self.resize(1200, 800)
        self.setAcceptDrops(True)

        self.project = Project()
        self.cache = PageCache()
        self.spread = SpreadWidget(self.cache)
        self.scroll = BookScrollArea(self.spread)
        self.scroll.setWidget(self.spread)
        self.spread.setScrollArea(self.scroll)
        self.setCentralWidget(self.scroll)

        self._make_actions()
        self._make_menus()
        self._make_toolbar()
        self._make_statusbar()

        self.spread.zoomRequested.connect(self._zoom_step)

    # ---------- UI setup -----------------------------------------------

    def _make_actions(self) -> None:
        def act(text, shortcut, slot, checkable=False, checked=False):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(slot)
            a.setCheckable(checkable)
            a.setChecked(checked)
            self.addAction(a)
            return a

        self.act_open = act("&Open directory…", "Ctrl+O", self.open_directory)
        self.act_quit = act("&Quit", "Ctrl+Q", self.close)

        self.act_new_project = act("&New project", "Ctrl+N", self.new_project)
        self.act_open_project = act("Open pro&ject…", "Ctrl+Shift+O", self.open_project)
        self.act_save_project = act("&Save project", "Ctrl+S", self.save_project)
        self.act_save_project_as = act("Save project &as…", "Ctrl+Shift+S", self.save_project_as)
        self.act_edit_project = act("&Edit project…", "Ctrl+E", self.edit_project)

        self.act_confirm = act("&Confirm pair", "C", self.confirm_and_advance)
        self.act_unconfirm = act("&Unconfirm pair", "U", self.unconfirm_pair)
        self.act_skip_left = act("Skip &left page", "Z", lambda: self.toggle_skip(0))
        self.act_skip_right = act("Skip ri&ght page", "X", lambda: self.toggle_skip(1))
        self.act_next_unconfirmed = act("Next unconfirmed pair", "N", self.next_unconfirmed)
        self.act_flag = act("&Flag for review", "F", self.toggle_flag)
        self.act_next_flagged = act("Next flagged spread", "G", self.next_flagged)
        self.act_blank_before = act(
            "Insert blank &before", "I", lambda: self.toggle_blank(True)
        )
        self.act_blank_after = act(
            "Insert blank &after", "Shift+I", lambda: self.toggle_blank(False)
        )

        self.act_next = act("Next spread", "Right", lambda: self._turn(1))
        self.act_next.setShortcuts([QKeySequence("Right"), QKeySequence("Space"), QKeySequence("PgDown")])
        self.act_prev = act("Previous spread", "Left", lambda: self._turn(-1))
        self.act_prev.setShortcuts([QKeySequence("Left"), QKeySequence("PgUp")])
        self.act_first = act("First spread", "Home", lambda: self._goto(0))
        self.act_last = act("Last spread", "End", lambda: self._goto(10**9))

        self.act_zoom_in = act("Zoom &In", "Ctrl+=", lambda: self._zoom_step(1.25))
        self.act_zoom_in.setShortcut(QKeySequence("Ctrl+="))
        self.act_zoom_out = act("Zoom &Out", "Ctrl+-", lambda: self._zoom_step(0.8))
        self.act_zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        self.act_zoom_reset = act("Reset zoom", "Ctrl+0", lambda: self._set_zoom(1.0))
        self.act_zoom_reset.setShortcut(QKeySequence("Ctrl+0"))
        self.act_goto = act("&Go to page…", "Ctrl+G", self.go_to_page)
        self.act_goto.setShortcut(QKeySequence("Ctrl+G"))

        self.act_fit_page = act("Fit &page", "W", lambda: self._set_fit("page"), True, True)
        self.act_fit_page.setShortcut(QKeySequence("W"))
        self.act_fit_width = act("Fit &width", "Shift+W", lambda: self._set_fit("width"), True)
        self.act_fit_width.setShortcut(QKeySequence("Shift+W"))
        group = QActionGroup(self)
        group.addAction(self.act_fit_page)
        group.addAction(self.act_fit_width)

        self.act_rtl = act("&Reverse (read from the back)", "R", self._toggle_rtl, True, False)

    def _make_menus(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        file_menu.addAction(self.act_new_project)
        file_menu.addAction(self.act_open_project)
        file_menu.addAction(self.act_open)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save_project)
        file_menu.addAction(self.act_save_project_as)
        file_menu.addAction(self.act_edit_project)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        go_menu = mb.addMenu("&Go")
        go_menu.addAction(self.act_next)
        go_menu.addAction(self.act_prev)
        go_menu.addAction(self.act_first)
        go_menu.addAction(self.act_last)
        go_menu.addSeparator()
        go_menu.addAction(self.act_goto)

        view_menu = mb.addMenu("&View")
        view_menu.addAction(self.act_zoom_in)
        view_menu.addAction(self.act_zoom_out)
        view_menu.addAction(self.act_zoom_reset)
        view_menu.addSeparator()
        view_menu.addAction(self.act_fit_page)
        view_menu.addAction(self.act_fit_width)
        view_menu.addAction(self.act_rtl)

        review_menu = mb.addMenu("&Review")
        review_menu.addAction(self.act_confirm)
        review_menu.addAction(self.act_unconfirm)
        review_menu.addAction(self.act_skip_left)
        review_menu.addAction(self.act_skip_right)
        review_menu.addAction(self.act_next_unconfirmed)
        review_menu.addSeparator()
        review_menu.addAction(self.act_blank_before)
        review_menu.addAction(self.act_blank_after)
        review_menu.addSeparator()
        review_menu.addAction(self.act_flag)
        review_menu.addAction(self.act_next_flagged)
        self.act_rtl.setShortcut(QKeySequence("R"))

    def _make_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        style = self.style()
        self.act_open.setIcon(style.standardIcon(QStyle.SP_DialogOpenButton))
        tb.addAction(self.act_open)
        self.act_prev.setIcon(style.standardIcon(QStyle.SP_ArrowBack))
        tb.addAction(self.act_prev)
        self.act_next.setIcon(style.standardIcon(QStyle.SP_ArrowForward))
        tb.addAction(self.act_next)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setPageStep(1)
        self.slider.setFixedWidth(260)
        self.slider.sliderMoved.connect(self._goto)
        tb.addWidget(self.slider)

        tb.addAction(self.act_zoom_out)
        tb.addAction(self.act_zoom_in)
        tb.addSeparator()
        tb.addAction(self.act_fit_page)
        tb.addAction(self.act_fit_width)
        tb.addAction(self.act_rtl)
        tb.addSeparator()
        self.act_confirm.setIcon(style.standardIcon(QStyle.SP_DialogApplyButton))
        tb.addAction(self.act_confirm)
        self.act_skip_left.setIcon(style.standardIcon(QStyle.SP_DialogResetButton))
        tb.addAction(self.act_skip_left)
        tb.addAction(self.act_next_unconfirmed)

    def _make_statusbar(self) -> None:
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("No book open")

    # ---------- slots ---------------------------------------------------

    # ---------- project / book management --------------------------------

    def _collect_pages(self) -> tuple[list[Path | None], list[str]]:
        """Concatenate pages from all project directories, minus skipped ones,
        with blank placeholders inserted at the recorded spots."""
        pages: list[Path | None] = []
        keys: list[str] = []
        for d in self.project.directories:
            for p in collect_images(d):
                key = self.project.key_for(d, p)
                if key in self.project.skip:
                    continue
                if f"blank:{key}" in self.project.blanks:
                    pages.append(None)
                    keys.append(f"blank:{key}")
                pages.append(p)
                keys.append(key)
        if "blank:__end__" in self.project.blanks:
            pages.append(None)
            keys.append("blank:__end__")
        return pages, keys

    def reload_book(self, anchor_key: str | None = None) -> None:
        """Rebuild the page list from the project; keep position via anchor_key."""
        # Keep the project's reverse state in sync with the UI (it is restored
        # from the JSON before this is called when a project is opened).
        self.project.reverse = self.act_rtl.isChecked()
        self.act_rtl.setChecked(self.project.reverse)
        self.spread.set_rtl(self.project.reverse)
        pages, keys = self._collect_pages()
        self.spread.set_book(pages, keys)
        self.spread.set_confirmed(self.project.confirmed)
        self.spread.set_flagged(self.project.flagged)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, self.spread.spread_count() - 1))
        self.slider.blockSignals(False)
        if anchor_key is not None and keys:
            for i in range(self.spread.spread_count()):
                spread = self.spread._spread_at(i)
                if spread and any(keys[j] == anchor_key for j in spread):
                    self._goto(i)
                    return
        self._goto(min(self.spread.spread_index(), max(0, self.spread.spread_count() - 1)))

    def _update_title(self) -> None:
        pages, _ = self._collect_pages()
        n_dirs = len(self.project.directories)
        self.setWindowTitle(
            f"Book Reader — {self.project.title()} "
            f"({n_dirs} dir{'s' if n_dirs != 1 else ''}, {len(pages)} pages)"
        )

    def open_directory(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open directory of pages", str(Path.cwd()))
        if d:
            self.project = Project()
            self.project.directories = [Path(d)]
            self.act_rtl.setChecked(False)   # fresh implicit project starts forward
            self.reload_book()

    def load_directory(self, directory: Path) -> None:
        self.project = Project()
        self.project.directories = [directory]
        self.act_rtl.setChecked(False)
        self.reload_book()

    def new_project(self) -> None:
        self.project = Project()
        self.act_rtl.setChecked(False)
        self.reload_book()

    def open_project(self) -> None:
        start = str(self.project.path.parent) if self.project.path else str(Path.cwd())
        f, _ = QFileDialog.getOpenFileName(
            self, "Open project", start, "Book Reader project (*.json)"
        )
        if f:
            self.open_project_path(Path(f))

    def open_project_path(self, path: Path) -> None:
        try:
            self.project.load(path)
        except (OSError, json.JSONDecodeError) as e:
            self.status.showMessage(f"Could not load project: {e}", 8000)
            return
        self.act_rtl.setChecked(self.project.reverse)   # restore saved direction
        self.reload_book()

    def save_project(self) -> None:
        if self.project.path is None:
            self.save_project_as()
            return
        try:
            self.project.save()
        except OSError as e:
            self.status.showMessage(f"Could not save project: {e}", 8000)
            return
        self._update_title()
        self.status.showMessage(f"Project saved to {self.project.path}", 5000)

    def save_project_as(self) -> None:
        start = str(self.project.path) if self.project.path else str(Path.cwd() / "book.json")
        f, _ = QFileDialog.getSaveFileName(
            self, "Save project", start, "Book Reader project (*.json)"
        )
        if not f:
            return
        if not f.endswith(".json"):
            f += ".json"
        try:
            self.project.save(Path(f))
        except OSError as e:
            self.status.showMessage(f"Could not save project: {e}", 8000)
            return
        self._update_title()
        self.status.showMessage(f"Project saved to {self.project.path}", 5000)

    def edit_project(self) -> None:
        if not self.project.directories:
            self.status.showMessage("Open a directory or project first", 5000)
            return
        anchor = self.spread.current_page_keys()
        dialog = ProjectDialog(self.project, self)
        if dialog.exec() != QDialog.Accepted:
            return
        base = self.project.path.parent if self.project.path else Path.cwd()
        self.project.directories = [
            Path(d) if Path(d).is_absolute() else base / d for d in dialog.directories()
        ]
        self.project.skip = set(dialog.skip())
        self.reload_book(anchor_key=anchor[0] if anchor else None)

    # ---------- pair confirmation / skipping -------------------------------

    def confirm_and_advance(self) -> None:
        keys = self.spread.current_page_keys()
        if len(keys) != 2:
            self.status.showMessage("Only two-page spreads can be confirmed", 5000)
            return
        pair = frozenset(keys)
        if pair not in self.project.confirmed:
            self.project.confirmed.add(pair)
            self.spread.set_confirmed(self.project.confirmed)
        # Move on to the next unconfirmed pair (in the current reading order).
        self.next_unconfirmed()

    def unconfirm_pair(self) -> None:
        keys = self.spread.current_page_keys()
        if len(keys) != 2:
            self.status.showMessage("Only two-page spreads can be unconfirmed", 5000)
            return
        self.project.confirmed.discard(frozenset(keys))
        self.spread.set_confirmed(self.project.confirmed)
        self._update_status()

    def toggle_skip(self, side: int) -> None:
        """Toggle skipping of the shown page: side 0 = left, 1 = right."""
        keys = self.spread.visible_page_keys()
        if not keys:
            return
        idx = min(side, len(keys) - 1)
        key = keys[idx]
        partner = keys[1 - idx] if len(keys) == 2 else None
        anchor = None
        if key.startswith("blank:"):
            self.project.blanks.discard(key)   # skipping a blank removes it
        elif key in self.project.skip:
            self.project.skip.discard(key)
            anchor = key          # it comes back; keep it on screen
        else:
            self.project.skip.add(key)
            if partner is not None:
                anchor = partner  # its partner stays put and re-pairs
            else:
                nxt = self.spread._spread_at(self.spread.spread_index() + 1)
                if nxt:
                    anchor = self.spread._keys[nxt[0]]
        self.reload_book(anchor_key=anchor)

    def toggle_blank(self, before: bool) -> None:
        """Insert (or remove) a blank page at the current spread boundary.

        ``before``: blank goes immediately before the spread's first page
        (file order); otherwise immediately after its last page.
        """
        spread = self.spread._spread_at(self.spread.spread_index())
        if not spread:
            return
        keys = self.spread._keys
        if before:
            blank_key = f"blank:{keys[spread[0]]}"
        else:
            after = spread[-1] + 1
            anchor_key = keys[after] if after < len(keys) else "__end__"
            blank_key = f"blank:{anchor_key}"
        if blank_key in self.project.blanks:
            self.project.blanks.discard(blank_key)
        else:
            self.project.blanks.add(blank_key)
        # Keep the current spread's first real page on screen.
        anchor = next((k for k in self.spread.current_page_keys() if not k.startswith("blank:")), None)
        self.reload_book(anchor_key=anchor)

    def next_unconfirmed(self) -> None:
        count = self.spread.spread_count()
        if count == 0:
            return
        for off in range(1, count + 1):
            i = (self.spread.spread_index() + off) % count
            spread = self.spread._spread_at(i)
            if not spread or len(spread) != 2:
                continue
            pair = frozenset(self.spread._keys[j] for j in spread)
            if pair not in self.project.confirmed:
                self._goto(i)
                return
        self.status.showMessage("All pairs confirmed ✓", 5000)

    def toggle_flag(self) -> None:
        keys = self.spread.current_page_keys()
        if not keys:
            return
        flag = frozenset(keys)
        if flag in self.project.flagged:
            self.project.flagged.discard(flag)
        else:
            self.project.flagged.add(flag)
        self.spread.set_flagged(self.project.flagged)
        self._update_status()

    def next_flagged(self) -> None:
        count = self.spread.spread_count()
        if count == 0:
            return
        for off in range(1, count + 1):
            i = (self.spread.spread_index() + off) % count
            spread = self.spread._spread_at(i)
            if not spread:
                continue
            if frozenset(self.spread._keys[j] for j in spread) in self.project.flagged:
                self._goto(i)
                return
        self.status.showMessage("No flagged spreads", 5000)

    def _turn(self, direction: int) -> None:
        if self.spread.spread_count() == 0:
            return
        step = -direction if self.act_rtl.isChecked() else direction
        self._goto(self.spread.spread_index() + step)

    def _goto(self, index: int) -> None:
        self.spread.set_spread_index(index)
        self.slider.blockSignals(True)
        self.slider.setValue(self.spread.spread_index())
        self.slider.blockSignals(False)
        self._reset_scroll()
        self._update_status()

    def _reset_scroll(self) -> None:
        for bar in (self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()):
            bar.setValue(0)

    def _zoom_step(self, factor: float) -> None:
        self._set_zoom(self.spread.zoom() * factor)

    def _set_zoom(self, zoom: float) -> None:
        old = self.spread.zoom()
        self.spread.set_zoom(zoom)
        ratio = zoom / old
        for bar in (self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()):
            center = bar.value() + bar.pageStep() / 2
            bar.setValue(int(center * ratio - bar.pageStep() / 2))

    def _set_fit(self, mode: str) -> None:
        self.spread.set_fit_mode(mode)
        self._reset_scroll()

    def _toggle_rtl(self) -> None:
        self.spread.set_rtl(self.act_rtl.isChecked())
        self.project.reverse = self.act_rtl.isChecked()
        # Start from the beginning of the current reading order:
        # the cover forward, the last page in reverse.
        self._goto(0)
        self._update_status()

    def go_to_page(self) -> None:
        if not self.spread._pages:
            return
        pages = self.spread.current_page_numbers()
        total = len(self.spread._pages)
        num, ok = QInputDialog.getInt(self, "Go to page", f"Page number (1–{total}):", pages[0], 1, total)
        if not ok:
            return
        target = num - 1
        for i in range(self.spread.spread_count()):
            spread = self.spread._spread_at(i)
            if spread is not None and target in spread:
                self._goto(i)
                return

    def _update_status(self) -> None:
        n = self.spread.spread_count()
        if n == 0:
            self.status.showMessage("No book open")
            self.act_confirm.setEnabled(False)
            self.act_skip_left.setEnabled(False)
            self.act_skip_right.setEnabled(False)
            return
        pages = self.spread.current_page_numbers()
        mode = "reverse (from the back)" if self.act_rtl.isChecked() else "forward"
        confirmed = self.spread.is_current_pair_confirmed()
        self.act_confirm.setEnabled(len(pages) == 2)
        self.act_skip_left.setEnabled(True)
        self.act_skip_right.setEnabled(True)
        mark = "   ·   ✓ confirmed" if confirmed else ""
        if self.spread.is_current_spread_flagged():
            mark += "   ·   ⚑ flagged"
        self.status.showMessage(
            f"Spread {self.spread.spread_index() + 1} / {n}   ·   page "
            f"{'–'.join(str(p) for p in pages)}   ·   {mode}   ·   "
            f"{len(self.project.confirmed)} pairs confirmed{mark}   ·   "
            f"{len(self.project.flagged)} flagged"
        )

    # ---------- drag & drop ----------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() == ".json":
                self.open_project_path(path)
                return
            if path.is_dir():
                self.load_directory(path)
                return
            if path.is_file():
                self.load_directory(path.parent)
                return


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Book Reader")
    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1])
        if arg.suffix.lower() == ".json":
            window.open_project_path(arg)
        else:
            window.load_directory(arg)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())