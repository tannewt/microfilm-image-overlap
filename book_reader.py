#!/usr/bin/env python3
"""Book-style two-page image viewer, like archive.org's BookReader.

Pages are shown two at a time as spreads; the first (cover) page and the
last page are shown alone, like a real book.

Projects are stored as JSON and hold the ordered list of page directories,
the pages to skip (duplicates), and the confirmed page pairs:

    {
        "version": 1,
        "directories": ["pages/P81-4720", "pages/P81-4721"],
        "plans_dirs": ["plans/P81-4720", ""],
        "skip": ["page005-scans011-011.png"],
        "confirmed_pairs": [["a.png", "b.png"], ...],
        "flagged_spreads": [["a.png", "b.png"], ...],
        "page_roles": {"page009-....png": "front",   # front | left | right | back
                       ...},
        "page_rotations": {"page012-....png": 90},   # degrees, clockwise
        "reverse": false                            # reverse (from the back) mode
    }

Marked pages shape the spreads: ``front`` pages get a blank before (solo
on the right half, like a cover), ``back`` pages get a blank after (solo
on the left half), and a ``left`` page pairs with a ``right`` page.

Usage:
    python book_reader.py [directory-or-project.json]

Opening a directory of pages (from the command line or Ctrl+O) creates a
project file named after it next to it (<directory>.json) unless one
already exists -- in that case it is loaded and the book resumes from its
saved state. All state autosaves to the project file as you work.

Controls:
    Ctrl+O            open a directory of page images
    Ctrl+N            new (empty) project
    Ctrl+Shift+O      open a project file
    Ctrl+S / Shift+Ctrl+S  save project / save as
                      (all state autosaves to the project file whenever it
                      changes, once the project has been saved once)
    Ctrl+E            edit project (directories, skip list)
    C                 confirm the pair, then jump to the next unconfirmed pair
    U                 unconfirm the showing pair (toggle off only)
    Z / X             skip the left / right shown page (toggle)
    N                 jump to the next unconfirmed pair
    F                 flag the spread for review (toggle)
    G                 jump to the next flagged spread
    Q / E             mark the left / right shown page as front (blank
                      before), back (blank after), left or right — or clear
                      the mark (menu)
    S                 mark the shown spread: left page back, right page front
    J / K             rotate the left / right shown page 90° clockwise
                      (display only; wraps back to 0)
    Right / Space     next spread (Left in reverse mode)
    Left              previous spread (Right in reverse mode)
    PageDown / Up     next / previous spread
    Home / End        first / last spread
    Ctrl+= / Ctrl+-   zoom in / out
    Ctrl+0            reset zoom
    Ctrl+G            go to page...
    W / Shift+W       fit page / fit width
    R                 toggle reverse (read from the back, last page first)
    [ / ]             toggle the left / right scan-contributor sidebar
    Ctrl+wheel        zoom
    drag              pan when zoomed in

The scan sidebars (for QA of composed output): when a shown page was
composited by compose_pages.py, its side of the book lists thumbnail
images of every scan that contributed to that page, with the region each
scan provided outlined in amber. Each page directory is mapped to the
plans/ directory holding its plan TOMLs -- guessed once when a book is
opened (conservatively: an ambiguous guess is left unset), or set or fixed
per directory in the project editor (Ctrl+E) and stored in the project's
plans_dirs. Click a thumbnail to open the full scan externally. Reverse
batches (plan_pages.py --reverse: dy < 0 as-shot) keep their film
coordinates in the scans' on-disk orientation too, so the outlined
regions are directly the contributed rows of the stored scans.
"""

from __future__ import annotations

import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import tomllib

from PySide6.QtCore import QRect, QObject, QPointF, QRectF, QRunnable, QSize, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QDesktopServices,
    QImage,
    QImageReader,
    QKeySequence,
    QPainter,
    QPixmap,
    QRegion,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QStatusBar,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
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


ROLE_FRONT = "front"   # solo on the right half, blank before (a cover)
ROLE_LEFT = "left"     # pairs with the page that follows it
ROLE_RIGHT = "right"   # pairs with the page that precedes it
ROLE_BACK = "back"     # solo on the left half, blank after


def build_marked_spreads(roles_by_index: list[str | None]) -> list[tuple[int | None, ...]]:
    """Build spreads from per-page side roles.

    Each spread is a 1–2 tuple of page indices, with ``None`` for a blank
    half. ``front`` pages get a blank spread half before them, ``back``
    pages one after; ``left``/``right`` pages pair up; unmarked pages pair
    sequentially (and a lone leftover finishes solo on the right half).
    """
    spreads: list[tuple[int | None, ...]] = []
    pending: int | None = None       # page awaiting its right-hand partner
    for i, role in enumerate(roles_by_index):
        if role == ROLE_FRONT:
            if pending is not None:
                spreads.append((pending,))       # previous page ends solo
                pending = None
            spreads.append((None, i))            # blank before the front
        elif role == ROLE_BACK:
            if pending is not None:
                spreads.append((pending,))
                pending = None
            spreads.append((i, None))            # blank after the back
        elif role == ROLE_RIGHT:
            if pending is not None:
                spreads.append((pending, i))     # pairs with the page before it
                pending = None
            else:
                spreads.append((None, i))        # nothing before: blank left
        elif role == ROLE_LEFT:
            if pending is not None:
                spreads.append((pending,))
            pending = i                            # waits for a right partner
        else:                                      # unmarked: pair sequentially
            if pending is not None:
                spreads.append((pending, i))
                pending = None
            else:
                pending = i
    if pending is not None:
        spreads.append((pending,))
    return spreads


class Project:
    """A book project: ordered page directories, pages to skip, confirmed pairs.

    JSON format (paths relative to the project file when possible):

        {
            "version": 1,
            "directories": ["pages/P81-4720", "pages/P81-4721"],
            "plans_dirs": ["plans/P81-4720", ""],   # parallel to directories;
                                                    # empty = unassigned
            "skip": ["page005-scans011-011.png"],
            "confirmed_pairs": [["a.png", "b.png"], ...]
        }
    """

    VERSION = 1

    def __init__(self) -> None:
        self.path: Path | None = None
        self.directories: list[Path] = []
        # Per page directory: the plans/ directory holding that batch's plan
        # TOMLs (for the scan-contributor sidebars), or None when unassigned.
        self.plans_dirs: list[Path | None] = []
        self.plans_dirs_saved: bool = True   # False for pre-plans_dirs projects
        self.skip: set[str] = set()
        self.confirmed: set[frozenset[str]] = set()
        self.flagged: set[frozenset[str]] = set()   # spreads marked for review
        # Page marks: key -> "front" | "left" | "right" | "back". front gets
        # a blank before (solo right half), back a blank after (solo left
        # half); left pairs with right.
        self.roles: dict[str, str] = {}
        # Page rotations: key -> degrees (0, 90, 180, 270), clockwise,
        # applied for display only (the files on disk are untouched).
        self.rotations: dict[str, int] = {}
        # "blank_pages" entries from pre-roles project files, converted to
        # roles on the next rebuild of the page list.
        self.legacy_blanks: set[str] = set()
        self.reverse: bool = False                   # reverse (read from the back) mode

    # ----- page identity ---------------------------------------------

    def _base(self) -> Path | None:
        return self.path.parent.resolve() if self.path else None

    def key_for(self, page_dir: Path, page: Path) -> str:
        """Stable identifier for a page: its full absolute path.

        Independent of the project file location, so saving or moving the
        project file never invalidates recorded state.
        """
        return str(page.resolve())

    # ----- id normalization -------------------------------------------

    @staticmethod
    def abs_id(key: str, project_dir: Path) -> str:
        """Convert a stored page id to a full absolute path.

        Older project files stored ids relative to the project file's
        directory (or mixed relative and absolute); resolve them so all
        forms match the ids key_for() generates.
        """
        p = Path(key)
        if not p.is_absolute():
            p = project_dir / p
        return str(p.resolve())

    # ----- serialization ----------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "directories": [str(d.resolve()) for d in self.directories],
            "plans_dirs": [str(d.resolve()) if d is not None else ""
                           for d in self.plans_dirs],
            "skip": sorted(self.skip),
            "confirmed_pairs": sorted([sorted(pair) for pair in self.confirmed]),
            "flagged_spreads": sorted([sorted(f) for f in self.flagged]),
            "page_roles": {k: self.roles[k] for k in sorted(self.roles)},
            "page_rotations": {
                k: self.rotations[k] for k in sorted(self.rotations)
            },
            "reverse": self.reverse,
        }

    def update_from_dict(self, data: dict, project_dir: Path) -> None:
        self.directories = [
            Path(p) if Path(p).is_absolute() else project_dir / p
            for p in data.get("directories", [])
        ]
        plans_entries = data.get("plans_dirs", [])
        self.plans_dirs = []
        for i in range(len(self.directories)):
            p = plans_entries[i] if i < len(plans_entries) else ""
            self.plans_dirs.append(
                None if not p
                else (Path(p) if Path(p).is_absolute() else project_dir / p)
            )
        # Projects saved before plans_dirs existed: the UI may fall back to
        # guessing the assignment on open.
        self.plans_dirs_saved = "plans_dirs" in data
        # Ids are full absolute paths; older project files stored ids
        # relative to the project file's directory.  Resolve them so all
        # forms match the ids key_for() generates.
        project_dir = project_dir.resolve()
        norm = lambda k: self.abs_id(k, project_dir)  # noqa: E731
        self.skip = {norm(k) for k in data.get("skip", [])}
        self.confirmed = {
            frozenset(norm(k) for k in pair)
            for pair in data.get("confirmed_pairs", [])
        }
        self.flagged = {
            frozenset(norm(k) for k in pair)
            for pair in data.get("flagged_spreads", [])
        }
        self.roles = {
            norm(k): r for k, r in (data.get("page_roles") or {}).items()
            if r in (ROLE_FRONT, ROLE_LEFT, ROLE_RIGHT, ROLE_BACK)
        }
        self.rotations = {}
        for k, r in (data.get("page_rotations") or {}).items():
            try:
                deg = int(r) % 360
            except (TypeError, ValueError):
                continue
            if deg:
                self.rotations[norm(k)] = deg
        # Migration: pre-roles projects stored blanks as "blank:<pagekey>"
        # (blank inserted before that page) or "blank:__end__" (after the
        # last page). They map to front/back marks; applied in reload_book
        # once the ordered page keys are known.
        self.legacy_blanks = {
            norm(b) for b in (data.get("blank_pages") or [])
            if "page_roles" not in data
        }
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Page ids are absolute paths, so saving to (or moving to) a new
        # location never invalidates recorded state.
        self.path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def title(self) -> str:
        return self.path.name if self.path else "(unsaved project)"

    @staticmethod
    def default_path_for(directory: Path) -> Path:
        """Project-file path for a book opened from a page-image directory:
        a sibling file named after it (<parent>/<book name>.json)."""
        d = directory.resolve()
        if d.parent == d:                       # filesystem root: avoid junk
            return Path.cwd() / "book.json"
        return d.parent / f"{d.name}.json"


class ProjectDialog(QDialog):
    """Editor for the project: page directories and the skip list."""

    def __init__(self, project: Project, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project")
        self.resize(640, 480)
        self._dirs = [str(d) for d in project.directories]
        self._plans = [
            str(project.plans_dirs[i]) if i < len(project.plans_dirs)
            and project.plans_dirs[i] is not None else ""
            for i in range(len(self._dirs))
        ]
        self._skip = sorted(project.skip)
        self._base = project._base()

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Page directories (in reading order), and the plans\n"
                                "directory each was composited from (empty = none, used by the\n"
                                "scan sidebars; double-click a cell to edit):"))
        self.dir_table = QTableWidget(len(self._dirs), 2, self)
        self.dir_table.setHorizontalHeaderLabels(["Page directory", "Plans directory"])
        self.dir_table.verticalHeader().setVisible(False)
        for r, (d, pd) in enumerate(zip(self._dirs, self._plans)):
            self.dir_table.setItem(r, 0, QTableWidgetItem(d))
            self.dir_table.setItem(r, 1, QTableWidgetItem(pd))
        header = self.dir_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.dir_table, 1)
        dir_buttons = QHBoxLayout()
        for text, slot in (
            ("Add…", self._add_dir),
            ("Remove", self._remove_dir),
            ("Up", lambda: self._move_dir(-1)),
            ("Down", lambda: self._move_dir(1)),
            ("Browse plans…", self._browse_plans),
            ("Guess plans", self._guess_plans),
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
            row = self.dir_table.rowCount()
            self.dir_table.insertRow(row)
            self.dir_table.setItem(row, 0, QTableWidgetItem(d))
            self.dir_table.setItem(row, 1, QTableWidgetItem(""))
            self.dir_table.setCurrentCell(row, 0)

    def _remove_dir(self) -> None:
        rows = sorted({i.row() for i in self.dir_table.selectedIndexes()} | ({self.dir_table.currentRow()} if self.dir_table.currentRow() >= 0 else set()),
                      reverse=True)
        for r in rows:
            self.dir_table.removeRow(r)

    def _move_dir(self, delta: int) -> None:
        row = self.dir_table.currentRow()
        new_row = row + delta
        if not (0 <= row < self.dir_table.rowCount()) \
                or not (0 <= new_row < self.dir_table.rowCount()):
            return
        for col in (0, 1):
            item = self.dir_table.takeItem(row, col)
            self.dir_table.setItem(row, col, self.dir_table.takeItem(new_row, col))
            self.dir_table.setItem(new_row, col, item)
        self.dir_table.setCurrentCell(new_row, self.dir_table.currentColumn())

    def _browse_plans(self) -> None:
        row = self.dir_table.currentRow()
        if row < 0:
            return
        parent = self.dir_table.item(row, 0)
        d = QFileDialog.getExistingDirectory(self, "Plans directory for this page directory",
                                            str(parent.text()) if parent else None)
        if d:
            self.dir_table.setItem(row, 1, QTableWidgetItem(d))

    def _guess_plans(self) -> None:
        for row in range(self.dir_table.rowCount()):
            cell = self.dir_table.item(row, 0)
            if cell is None:
                continue
            current = self.dir_table.item(row, 1)
            if current is not None and current.text().strip():
                continue          # explicitly set: leave it alone
            plans = guess_plans_dir(Path(cell.text()), self._guess_base())
            if plans is not None:
                self.dir_table.setItem(row, 1, QTableWidgetItem(str(plans)))

    def _guess_base(self) -> Path | None:
        return self._base

    def _remove_skip(self) -> None:
        for item in self.skip_list.selectedItems():
            self.skip_list.takeItem(self.skip_list.row(item))

    # ----- results -------------------------------------------------------

    def directories(self) -> list[str]:
        return [self.dir_table.item(r, 0).text()
                if self.dir_table.item(r, 0) else ""
                for r in range(self.dir_table.rowCount())]

    def plans_dirs(self) -> list[str]:
        return [self.dir_table.item(r, 1).text().strip()
                if self.dir_table.item(r, 1) else ""
                for r in range(self.dir_table.rowCount())]


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


        try:
            self.cache.imageLoaded.emit(self.path)
        except RuntimeError:
            pass  # cache (or app) already torn down


# --------------------------------------------------------------------- #
# Scan-contributor sidebars (QA): for each shown page, the scans that    #
# composited it, as thumbnails with their crop region outlined.          #
# ---------------------------------------------------------------------  #

SIDEBAR_W = 208        # px width of each sidebar
SIDEBAR_THUMB_W = 184  # px width of a scan thumbnail


def guess_plans_dir(page_dir: Path, base: Path | None) -> Path | None:
    """Best-effort plans/ directory for a page directory, or None.

    Only ever matches a SINGLE plans directory -- page labels repeat across
    batches (page001-scans000-000 exists in six plan dirs), so guessing must
    never fall back to a broad search that could pick a different batch's
    plan. If the mapping is ambiguous, None is returned and the user sets it
    in the project dialog (Ctrl+E, stored in the project's plans_dirs).

    Rules tried in order for pages/.../X relative to `base`:
      1. exact mirror: plans/.../X
      2. joined-key match: page dirs are often batch+subdir names flattened
         with dashes ("20260929-05671-6200" -> plans/20260929/05671-6200),
         so match the page path's dash-joined key against plans dirs' keys
         (exact, then unique/longest suffix extension like "X-flat").
    """
    page_dir = page_dir.resolve()
    rel_parts: list[str] = []
    if base is not None:
        try:
            rel_parts = list(page_dir.relative_to(base.resolve()).parts)
        except ValueError:
            rel_parts = []
    if not rel_parts:
        rel_parts = [page_dir.name]
    if not rel_parts:
        return None

    # Collect candidate plans roots: a plans/ dir beside the page dir or any
    # of its ancestors, beside `base`, and the repo's plans/ when the page
    # dir sits in a pages/ tree with plans/ nearby.
    roots: list[Path] = []
    for anc in [page_dir, *page_dir.parents][:5]:
        plans = anc / "plans"
        if plans.is_dir() and plans not in roots:
            roots.append(plans)

    def find_all_dirs(plans_root: Path) -> list[tuple[Path, str]]:
        out = []
        for p in plans_root.rglob("*"):
            if p.is_dir():
                out.append((p, "-".join(p.relative_to(plans_root).parts)))
        return out

    for plans_root in roots:
        dirs = find_all_dirs(plans_root)
        # 1) exact mirror of the full relative path (pages -> plans)
        if rel_parts[0] == "pages":
            cand = plans_root.joinpath(*rel_parts[1:])
            if cand.is_dir():
                return cand
        # 2) dash-joined key of the page dir vs plans dirs' keys. A leading
        # "pages" (the tree's bookkeeping component) doesn't take part.
        key_parts = rel_parts[1:] if rel_parts[0] == "pages" else rel_parts
        key = "-".join(key_parts)
        exact = [d for d, k in dirs if k == key]
        if len(exact) == 1:
            return exact[0]
        # e.g. keys "20260929-05671-6200" <-> dirs of the same name
        extended = sorted(
            ((len(k), d) for d, k in dirs
             if key and len(key) > len(k) and key.startswith(k)
             and key[len(k)] in "-_"),
            key=lambda t: -t[0],
        )
        if len(extended) == 1 or (len(extended) > 1
                                  and extended[0][0] > extended[1][0]):
            return extended[0][1]
        # and the reverse: page key shorter than the dir key, e.g. page dir
        # "4722-9113-09974" vs plans/20260924/4722-9113-09974
        tail = [d for d, k in dirs if key and k.endswith("-" + key)]
        if len(tail) == 1:
            return tail[0]
    return None


class PlanLibrary:
    """Finds and parses each page's plan TOML, with caching.

    ``info_for`` returns a dict:

        {"toml": Path|None, "kind": "page"|"gap_join"|"missing",
         "scans": [{"name", "path"|None, "crop"|None, "display_size"}],
         "missing": [scan names that could not be located],
         "error": str|None}

    The scans are the page's contributors (manual overrides applied, so a
    `skip = true` contributor is absent); ``crop`` is the region each scan
    provided in the scan's on-disk coordinates (the composited page shares
    the scans' reading orientation, so thumbnails show the scan as stored;
    for axis="x" plans the film axis runs along the scan's width, and for
    `reverse = true` (--reverse) plans the film coordinates already are the
    scans' on-disk rows), with ``display_size`` the scan's
    on-disk (width, height).
    Returns None when `plans_dir` is None (no plans directory assigned for
    the page's directory); a stub with toml=None when the plan file is just
    absent, so the UI can report each case precisely.
    """

    LIMIT = 256

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[str, str], dict | None] = OrderedDict()

    def info_for(self, page: Path, plans_dir: Path | None) -> dict | None:
        key = (str(page), str(plans_dir) if plans_dir else "")
        if key in self._cache:
            info = self._cache[key]
            self._cache.move_to_end(key)
            return info
        info = self._parse(page, plans_dir)
        self._cache[key] = info
        while len(self._cache) > self.LIMIT:
            self._cache.popitem(last=False)
        return info

    @staticmethod
    def _parse(page: Path, plans_dir: Path | None) -> dict | None:
        if plans_dir is None:
            return None
        toml_path = plans_dir / (page.stem + ".toml")
        if not toml_path.is_file():
            return {"toml": None, "kind": None, "scans": [],
                    "missing": [], "error": None}
        try:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as e:
            return {"toml": toml_path, "kind": None, "scans": [],
                    "missing": [], "error": f"could not read plan: {e}"}
        # Reuse the compositor's override logic + scan-name resolution; the
        # heavy cv2/numpy imports happen only when a plan is first parsed.
        try:
            from compose_pages import ScanResolver, apply_manual
            eff = apply_manual(data)
        except (SystemExit, Exception) as e:   # noqa: BLE001 -- show, don't die
            return {"toml": toml_path, "kind": None, "scans": [],
                    "missing": [], "error": f"bad plan: {e}"}
        kind = eff["kind"]
        axis = eff.get("axis", "y")
        scans, missing = [], []
        seen: set[str] = set()
        source_dir = data.get("page", {}).get("source_dir")
        scan_note = None
        if source_dir:
            scan_dir = Path(source_dir)
            if page.parent.name.endswith("-flat"):
                # A -flat page variant was composited from flat-twin scans
                # (source_dir/flat); prefer them so the thumbnails show
                # what actually produced the page.
                twin = scan_dir / "flat"
                if twin.is_dir():
                    scan_dir = twin
                    scan_note = "flat-twin scans"
            if not scan_dir.is_dir():
                scan_note = f"plan's scans dir ({scan_dir}) missing on disk"
                scan_dir = None
        else:
            scan_dir = None
            scan_note = "plan has no source_dir"
        resolver = ScanResolver(scan_dir) if scan_dir else None
        # (film window, contribs) pairs: for kind="page" the page's film
        # range covers all contributors; for a gap_join each section has its
        # own range and its own canvas (they stack, with red bands between).
        windows: list[tuple[list[float], list[dict]]] = []
        if kind == "page":
            film = eff.get("film")
            if film:
                windows.append((film, eff.get("contribs", [])))
        elif kind == "gap_join":
            for sec in eff.get("sections", []):
                film = sec.get("film")
                if film:
                    windows.append((film, sec.get("contribs") or []))

        scans, missing = [], []
        seen: set[str] = set()
        source_dir = data.get("page", {}).get("source_dir")
        scan_note = None
        if source_dir:
            scan_dir = Path(source_dir)
            if page.parent.name.endswith("-flat"):
                # A -flat page variant was composited from flat-twin scans
                # (source_dir/flat); prefer them so the thumbnails show
                # what actually produced the page.
                twin = scan_dir / "flat"
                if twin.is_dir():
                    scan_dir = twin
                    scan_note = "flat-twin scans"
            if not scan_dir.is_dir():
                scan_note = f"plan's scans dir ({scan_dir}) missing on disk"
                scan_dir = None
        else:
            scan_dir = None
            scan_note = "plan has no source_dir"
        resolver = ScanResolver(scan_dir) if scan_dir else None

        for film, contribs in windows:
            f_lo, f_hi = film[0], film[1]
            canvas_len = int(round(f_hi - f_lo))   # the page's film-axis span
            # Geometry pass, mirroring assemble_page exactly (including its
            # `_axis_bbox` reframe for axis="x", int() clamp bounds, first
            # surviving contributor as dx anchor, and the hard-paste's canvas
            # clipping), so the shown rectangles match what actually landed
            # on the composited page.
            pieces = []
            for c in contribs:
                bbox = c.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                top, bottom, left, right = (int(v) for v in bbox)
                if axis == "x":          # assembly frame: film axis = cols
                    top, bottom, left, right = left, right, top, bottom
                s = float(c.get("film_offset", 0.0))
                r_lo = max(f_lo - s, top)
                r_hi = min(f_hi - s, bottom)
                if r_hi - r_lo < 2:      # dropped by the compositor
                    continue
                pieces.append((c, int(r_lo), int(r_hi),
                               float(c.get("dx", 0.0)),
                               top, bottom, left, right))
            if not pieces:
                continue
            # dx anchor: x0 = each piece's dx delta from the first surviving
            # piece (only deltas matter; see assemble_page).
            base_dx = pieces[0][3]
            pieces = [(c, plo, phi, int(round(dx - base_dx)), t, b, l, r)
                      for (c, plo, phi, dx, t, b, l, r) in pieces]
            x_min = min(p[3] for p in pieces)
            x_max = max(p[3] + p[7] - p[6] + 1 for p in pieces)
            canvas_w = x_max - x_min
            for (c, p_lo, p_hi, x0, at_, ab, al, ar) in pieces:
                name = c.get("scan") or "?"
                if name in seen:      # one thumbnail per contributing scan
                    continue
                seen.add(name)
                path = None
                if resolver is not None:
                    try:
                        path = Path(resolver.resolve(name))
                    except SystemExit:
                        path = None
                if path is None:
                    missing.append(name)
                size = c.get("size")
                dsize = None
                if size and len(size) == 2:
                    sh, sw = int(size[0]), int(size[1])   # size = [rows, cols]
                    dsize = QSize(sw, sh)                 # scan as stored
                th_ = p_hi - p_lo + 1             # film-axis tile span
                tw_ = ar - al + 1                 # across-axis tile span
                y0 = int(round(float(c.get("film_offset", 0.0)) + p_lo - f_lo))
                cx0 = x0 - x_min
                # What survives the hard paste onto the canvas.
                ry0, ry1 = max(0, y0), min(canvas_len, y0 + th_)
                rx0, rx1 = max(0, cx0), min(canvas_w, cx0 + tw_)
                if ry1 <= ry0 or rx1 <= rx0:
                    continue                      # nothing of this piece shows
                # ...mapped back into the scan's on-disk coordinates: the
                # film axis is the scan's rows for axis="y" and its columns
                # for axis="x"; the across axis is the other one.
                r_top = p_lo + (ry0 - y0)
                r_bot = p_lo + (ry1 - 1 - y0)
                c_left = al + (rx0 - cx0)
                c_right = al + (rx1 - 1 - cx0)
                if axis == "x":
                    crop = QRect(r_top, c_left,
                                 r_bot - r_top + 1, c_right - c_left + 1)
                else:
                    crop = QRect(c_left, r_top,
                                 c_right - c_left + 1, r_bot - r_top + 1)
                scans.append({"name": name, "path": path, "crop": crop,
                              "display_size": dsize})
        return {"toml": toml_path, "kind": kind, "scans": scans,
                "missing": missing, "error": scan_note}


class ThumbLoader(QObject):
    """Background loader + small LRU cache of downscaled scan thumbnails.

    ``request`` returns a key immediately and starts a worker if the
    thumbnail isn't cached yet; when it's ready ``thumbReady`` fires (queued
    to the GUI thread) and the image is retrievable with ``get(key)``.
    """

    LIMIT = 48
    thumbReady = Signal(str, QImage)

    def __init__(self) -> None:
        super().__init__()
        self._cache: OrderedDict[str, QImage] = OrderedDict()
        self._pending: set[str] = set()
        # The worker emits thumbReady from its thread; this queued connection
        # runs _store on the GUI thread, where the cache belongs.
        self.thumbReady.connect(self._store)

    def get(self, key: str) -> QImage | None:
        img = self._cache.get(key)
        if img is not None:
            self._cache.move_to_end(key)
        return img

    def request(self, path: Path | str, width: int) -> str:
        path = str(path)
        key = f"{path}|{width}"
        if self.get(key) is not None:
            return key
        if key not in self._pending:
            self._pending.add(key)
            QThreadPool.globalInstance().start(
                _ThumbTask(self, key, path, width)
            )
        return key

    def _store(self, key: str, img: QImage) -> None:
        """Cache the finished thumbnail (runs on the GUI thread)."""
        self._pending.discard(key)
        if img.isNull():
            return                      # don't cache failures
        self._cache[key] = img
        self._cache.move_to_end(key)
        while len(self._cache) > self.LIMIT:
            self._cache.popitem(last=False)


class _ThumbTask(QRunnable):
    def __init__(self, loader: ThumbLoader, key: str, path: str, width: int) -> None:
        super().__init__()
        self.loader = loader
        self.key = key
        self.path = path
        self.width = width

    def run(self) -> None:
        reader = QImageReader(self.path)
        reader.setAutoTransform(True)
        img = reader.read()
        if not img.isNull():
            img = img.scaledToWidth(max(1, self.width), Qt.SmoothTransformation)
        try:
            self.loader.thumbReady.emit(self.key, img)
        except RuntimeError:
            pass  # loader (or app) already torn down


class ThumbCell(QLabel):
    """One scan thumbnail: the downscaled scan, its crop region outlined.
    Clicking opens the full scan in the system image viewer.
    """

    def __init__(
        self,
        loader: ThumbLoader,
        path: Path | None,
        name: str,
        crop: QRect | None,
        display_size: QSize | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._path = path
        self._name = name
        self._crop = crop
        self._display_size = display_size
        self._key = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(SIDEBAR_THUMB_W, 100)
        self.setStyleSheet(
            "background: #1e1e1e; border: 1px solid #3a3a3a; color: #666666;"
            "font-size: 11px;"
        )
        self.setCursor(Qt.PointingHandCursor)
        tip = name
        if path is not None:
            tip += f"\n{path}\n(click to open in the image viewer)"
        self.setToolTip(tip)
        if path is not None:
            self._key = loader.request(path, SIDEBAR_THUMB_W)
            img = loader.get(self._key)
            if img is not None:
                self._apply(img)
            else:
                self.setText("…")
            loader.thumbReady.connect(self._on_ready)
        else:
            self.setText("missing")

    def _on_ready(self, key: str, img: QImage) -> None:
        if key != self._key:
            return
        self._apply(img)

    def _apply(self, img: QImage) -> None:
        if img is None or img.isNull():
            self.setText("failed to load")
            return
        # Deep-copy the cached image before annotating: several cells can
        # hold the same cached QImage (a scan shared by two pages), and the
        # per-cell crop/shade annotations must never leak into the cache or
        # into a sibling cell's pixmap.
        pm = QPixmap.fromImage(img.copy())
        if self._crop is not None:
            scale = (pm.width() / self._display_size.width()
                     if self._display_size and self._display_size.width()
                     else 1.0)
            if self._display_size is None:
                # Without the full size, assume the crop fills the image.
                rect = QRect(0, 0, pm.width(), pm.height())
            else:
                rect = QRect(
                    int(self._crop.x() * scale), int(self._crop.y() * scale),
                    max(1, int(self._crop.width() * scale)),
                    max(1, int(self._crop.height() * scale)),
                )
            painter = QPainter(pm)
            if rect != QRect(0, 0, pm.width(), pm.height()):
                # Shade the parts of the scan that did NOT contribute (dim
                # maroon so it reads even over the scan's black frame bars).
                outside = QRegion(pm.rect()) - QRegion(rect)
                painter.setClipRegion(outside)
                painter.fillRect(pm.rect(), QColor(96, 10, 10, 150))
                painter.setClipRegion(QRegion(pm.rect()))
            # Outline the contributed region, just inside any edge it shares
            # with the scan's border, so all four edges stay visible (a box
            # spanning a full dimension must still show where it ends).
            if rect.left() <= 0:
                rect.setLeft(3)
            if rect.top() <= 0:
                rect.setTop(3)
            if rect.right() >= pm.width() - 1:
                rect.setRight(pm.width() - 4)
            if rect.bottom() >= pm.height() - 1:
                rect.setBottom(pm.height() - 4)
            pen = painter.pen()
            pen.setColor(QColor(232, 176, 36))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.end()
        self.setPixmap(pm)
        self.setMinimumSize(SIDEBAR_THUMB_W, int(pm.height()))
        self.setStyleSheet(
            "background: #1e1e1e; border: 1px solid #3a3a3a; font-size: 11px;"
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._path)))
            event.accept()
        else:
            super().mousePressEvent(event)


class ScanSidebar(QWidget):
    """Vertical strip listing the scans behind one side's shown page."""

    def __init__(self, loader: ThumbLoader, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loader = loader
        self.setFixedWidth(SIDEBAR_W)
        self.setStyleSheet(
            "ScanSidebar { background: #141414; }"
            "QLabel#sidebarHeader { color: #bbbbbb; font-size: 12px;"
            "   padding: 4px 2px 2px 6px; }"
            "QLabel#sidebarMsg { color: #777777; font-size: 11px; padding: 6px; }"
            "QLabel#sidebarName { color: #8a8a8a; font-size: 9px; padding: 0 2px;}"
            "QScrollArea { border: none; background: #141414; }"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        self.header = QLabel("", self)
        self.header.setWordWrap(True)
        self.header.setObjectName("sidebarHeader")
        lay.addWidget(self.header)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.viewport().setStyleSheet("background: #141414;")
        lay.addWidget(self.scroll, 1)
        self.scroll.setWidget(self._wrap(QLabel("no page")))

    @staticmethod
    def _wrap(inner: QWidget) -> QWidget:
        holder = QWidget()
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(3)
        lay.addWidget(inner)
        lay.addStretch(1)
        holder.setStyleSheet("background: #141414;")
        return holder

    def set_page(self, title: str, message: str | None, scans: list[dict]) -> None:
        """Rebuild the list: `title` in the header, then one ThumbCell per
        scan entry ({name, path, crop, display_size}), or `message` if empty."""
        self.header.setText(title)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(3)
        if message:
            msg = QLabel(message, body)
            msg.setWordWrap(True)
            msg.setObjectName("sidebarMsg")
            lay.addWidget(msg)
        metrics = self.header.fontMetrics()
        for scan in scans:
            name = metrics.elidedText(
                scan["name"], Qt.ElideMiddle, SIDEBAR_THUMB_W - 10
            )
            label = QLabel(name, body)
            label.setObjectName("sidebarName")
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            cell = ThumbCell(
                self._loader, scan["path"], scan["name"], scan["crop"],
                scan["display_size"], body,
            )
            lay.addWidget(cell)
            lay.addWidget(label)
        old = self.scroll.takeWidget()
        self.scroll.setWidget(self._wrap(body))
        if old is not None:
            old.deleteLater()


class SpreadWidget(QWidget):
    """Draws the current spread (one or two pages) scaled to fit the viewport."""

    zoomRequested = Signal(float)   # multiplicative zoom step, e.g. 1.25 / 0.8

    def __init__(self, cache: PageCache, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cache = cache
        cache.imageLoaded.connect(self._on_image_loaded)
        self._pages: list[Path] = []
        self._keys: list[str] = []
        self._spreads: list[tuple[int | None, ...]] = []
        self._spreads_back: list[tuple[int | None, ...]] = []
        self._roles: dict[str, str] = {}
        self._rotations: dict[str, int] = {}
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

    def set_book(self, pages: list[Path], keys: list[str] | None = None,
                 spreads: list[tuple[int | None, ...]] | None = None,
                 roles: dict[str, str] | None = None,
                 rotations: dict[str, int] | None = None) -> None:
        """Set the book. ``spreads`` holds page indices per spread, with
        ``None`` entries for the blank half of front/back spreads; when
        omitted, spreads are built by plain sequential pairing. ``rotations``
        maps page keys to display degrees (clockwise, 0/90/180/270)."""
        self._pages = pages
        self._keys = list(keys) if keys is not None else [str(p) for p in pages]
        self._spreads = list(spreads) if spreads is not None \
            else build_spreads(len(pages))
        # Reverse mode reads the mirror image of the book: spreads in the
        # opposite order, each one flipped left-to-right.
        self._spreads_back = [tuple(reversed(s)) for s in reversed(self._spreads)]
        self._roles = dict(roles) if roles else {}
        self._rotations = dict(rotations) if rotations else {}
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
        return tuple(i + 1 for i in spread if i is not None)

    def current_page_keys(self) -> tuple[str, ...]:
        spread = self._spread_at(self._index)
        if spread is None:
            return ()
        return tuple(self._keys[i] for i in spread if i is not None)

    def visible_page_keys(self) -> tuple[str, ...]:
        """Real pages' keys in on-screen order: left page first, right second
        (blank halves are skipped)."""
        return tuple(self._keys[i] for i in self._visible_entries() if i is not None)

    def visible_pages(self) -> tuple[Path | None, ...]:
        """Page paths (None = blank half) in on-screen order."""
        return tuple(None if i is None else self._pages[i]
                     for i in self._visible_entries())

    def visible_sides(self) -> list[tuple[Path | None, int | None, int | None]]:
        """One (page, page_number, book_index) per on-screen side; a blank
        half of a front/back spread shows as (None, None, None)."""
        return [
            (None, None, None) if i is None
            else (self._pages[i], i + 1, i)
            for i in self._visible_entries()
        ]

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

    def _visible_entries(self) -> list[int | None]:
        """Spread entries (page index, or None = blank half), on-screen order."""
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
        aspects = []
        rotated = []
        for entry in entries:
            if entry is None:            # blank half (front/back spread)
                images.append(None)
                aspects.append(BLANK_ASPECT)
                rotated.append(0)
                continue
            img = self._cache.get_sync(str(self._pages[entry]))
            images.append(img)
            rot = self._rotations.get(self._keys[entry], 0)
            if img is not None and img.height():
                aspect = img.width() / img.height()
                if rot % 180:          # 90/270: width and height swap
                    aspect = img.height() / img.width()
                aspects.append(aspect)
            else:
                aspects.append(1 / 0.65 if rot % 180 else 0.65)
            rotated.append(rot)
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
            and all(e is not None for e in entries)
            and frozenset(self._keys[i] for i in entries) in self._confirmed
        )
        flagged = frozenset(self._keys[i] for i in entries if i is not None) \
            in self._flagged
        suffix = ("  ✓" if pair_confirmed else "") + ("  ⚑" if flagged else "")
        for i, entry in enumerate(entries):
            rect = QRectF(xs[i], MARGIN, widths[i] * scale, page_h)
            blank = entry is None
            role = None if blank else self._roles.get(self._keys[entry])
            rot = 0 if blank else rotated[i]
            pixmap = None
            img = images[i]
            if not blank and img is not None:
                if rot % 180:
                    # Scale to the swapped (rotated) rectangle, then turn the
                    # pixmap to fill rect exactly.
                    scaled = img.scaled(
                        QSize(int(rect.height()), int(rect.width())),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation,
                    )
                    pixmap = QPixmap.fromImage(scaled).transformed(
                        QTransform().rotate(rot)
                    )
                else:
                    target = QSize(int(rect.width()), int(rect.height()))
                    pixmap = QPixmap.fromImage(img.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            caption = "" if blank else \
                f"{entry + 1} / {n_pages}" \
                + (f"  ·{role}" if role else "") \
                + (f"  ⟳{rot}" if rot else "") \
                + suffix
            self._items.append((rect, pixmap, caption, blank))

        self.update()

    def _on_image_loaded(self, path: str) -> None:
        # A prefetched neighbor arrived; repaint only if it is visible now.
        if any(
            entry is not None and str(self._pages[entry]) == path
            for entry in self._visible_entries()
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
                    str(self._pages[i]) for i in spread if i is not None
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

        # Scan-contributor sidebars (QA), one per side of the book.
        self.plans = PlanLibrary()
        self.thumbs = ThumbLoader()
        self.left_sidebar = ScanSidebar(self.thumbs)
        self.right_sidebar = ScanSidebar(self.thumbs)
        sidebars = QWidget(self)
        side_lay = QHBoxLayout(sidebars)
        side_lay.setContentsMargins(0, 0, 0, 0)
        side_lay.setSpacing(0)
        side_lay.addWidget(self.left_sidebar)
        side_lay.addWidget(self.scroll, 1)
        side_lay.addWidget(self.right_sidebar)
        self.setCentralWidget(sidebars)
        self.left_sidebar.hide()
        self.right_sidebar.hide()

        self._make_actions()
        self._make_menus()
        self._make_toolbar()
        self._make_statusbar()

        # Debounced autosave: state changes save the project on their own
        # once it has a file.
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(800)
        self._autosave_timer.timeout.connect(self._autosave_now)

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
        self.act_mark_front = act(
            "Mark &front (blank before)", "Q", lambda: self.mark_page(0)
        )
        self.act_mark_other = act(
            "Mark right / other &page", "E", lambda: self.mark_page(1)
        )
        self.act_mark_sides = act(
            "Mark sides: left &back, right front", "S", self.mark_sides
        )
        self.act_rot_left = act(
            "Rotate &left page", "J", lambda: self.rotate_page(0)
        )
        self.act_rot_right = act(
            "Rotate ri&ght page", "K", lambda: self.rotate_page(1)
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

        self.act_sidebar_left = act(
            "Left scan sidebar", "[", lambda: self._toggle_sidebar(0), True, False
        )
        self.act_sidebar_left.setShortcut(QKeySequence("["))
        self.act_sidebar_right = act(
            "Right scan sidebar", "]", lambda: self._toggle_sidebar(1), True, False
        )
        self.act_sidebar_right.setShortcut(QKeySequence("]"))

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
        view_menu.addSeparator()
        view_menu.addAction(self.act_sidebar_left)
        view_menu.addAction(self.act_sidebar_right)

        review_menu = mb.addMenu("&Review")
        review_menu.addAction(self.act_confirm)
        review_menu.addAction(self.act_unconfirm)
        review_menu.addAction(self.act_skip_left)
        review_menu.addAction(self.act_skip_right)
        review_menu.addAction(self.act_next_unconfirmed)
        review_menu.addSeparator()
        review_menu.addAction(self.act_mark_front)
        review_menu.addAction(self.act_mark_other)
        review_menu.addAction(self.act_mark_sides)
        review_menu.addSeparator()
        review_menu.addAction(self.act_rot_left)
        review_menu.addAction(self.act_rot_right)
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

    def _collect_pages(self) -> tuple[list[Path], list[str]]:
        """Concatenate pages from all project directories, minus skipped ones."""
        pages: list[Path] = []
        keys: list[str] = []
        page_plans: list[Path | None] = []   # parallel: plans dir per page
        for di, d in enumerate(self.project.directories):
            plans_dir = (self.project.plans_dirs[di]
                         if di < len(self.project.plans_dirs) else None)
            for p in collect_images(d):
                key = self.project.key_for(d, p)
                if key in self.project.skip:
                    continue
                pages.append(p)
                keys.append(key)
                page_plans.append(plans_dir)
        self._page_plans = page_plans
        return pages, keys

    def _build_spreads(self, keys: list[str]) -> list[tuple[int | None, ...]]:
        """Spreads for the book: role-driven when any page is marked, plain
        sequential pairing otherwise."""
        roles = [self.project.roles.get(k) for k in keys]
        if any(roles):
            return build_marked_spreads(roles)
        return build_spreads(len(keys))

    def _apply_legacy_blank_marks(self, keys: list[str]) -> None:
        """Convert pre-roles "blank_pages" marks into page roles (called once
        per project load): a blank recorded before page X becomes a front mark
        on X, and one recorded at the very end becomes a back mark on the last
        page."""
        if not self.project.legacy_blanks:
            return
        for entry in sorted(self.project.legacy_blanks):
            if entry == "blank:__end__":
                if keys:
                    self.project.roles[keys[-1]] = ROLE_BACK
            elif entry.startswith("blank:") and entry[6:] in set(keys):
                self.project.roles[entry[6:]] = ROLE_FRONT
        self.project.legacy_blanks = set()

    def reload_book(self, anchor_key: str | None = None) -> None:
        """Rebuild the page list from the project; keep position via anchor_key."""
        # Keep the project's reverse state in sync with the UI (it is restored
        # from the JSON before this is called when a project is opened).
        self.project.reverse = self.act_rtl.isChecked()
        self.act_rtl.setChecked(self.project.reverse)
        self.spread.set_rtl(self.project.reverse)
        pages, keys = self._collect_pages()
        self._apply_legacy_blank_marks(keys)
        self.spread.set_book(pages, keys, self._build_spreads(keys),
                             self.project.roles, self.project.rotations)
        self.spread.set_confirmed(self.project.confirmed)
        self.spread.set_flagged(self.project.flagged)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, self.spread.spread_count() - 1))
        self.slider.blockSignals(False)
        self._update_title()
        if anchor_key is not None and keys:
            for i in range(self.spread.spread_count()):
                spread = self.spread._spread_at(i)
                if spread and any(j is not None and keys[j] == anchor_key
                                  for j in spread):
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
            self.load_directory(Path(d))

    def load_directory(self, directory: Path) -> None:
        # No project file given: start one next to the pages so state
        # autosaves from the first change on.  If one already exists there
        # (same book opened again), load it instead and resume where you
        # left off.
        proj_path = Project.default_path_for(directory)
        if proj_path.exists():
            self.open_project_path(proj_path)
            return
        self.project = Project()
        self.project.directories = [directory]
        self.project.plans_dirs = [guess_plans_dir(Path(directory), None)]
        self.act_rtl.setChecked(False)   # fresh implicit project starts forward
        self.project.path = proj_path
        try:
            self.project.save()
        except OSError as e:
            self.project.path = None
            self.status.showMessage(
                f"Could not create project file: {e}; use Ctrl+S to pick "
                "a location", 8000)
        else:
            self.status.showMessage(
                f"Project file {self.project.path} — state autosaves", 5000)
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
        # Plans-dir assignments: never re-guess what a project file saved
        # explicitly (even empty = user unset it in the editor). For pre-
        # plans_dirs projects, guess one per page directory now (conservatively:
        # an ambiguous guess stays None, fixable in the editor, Ctrl+E).
        if not self.project.plans_dirs_saved:
            self.project.plans_dirs = [
                guess_plans_dir(d, self.project._base())
                if d is not None and d.is_dir() else None
                for d in self.project.directories
            ]
        self.act_rtl.setChecked(self.project.reverse)   # restore saved direction
        self.reload_book()

    def save_project(self) -> None:
        if self.project.path is None:
            self.save_project_as()
            return
        self._autosave_timer.stop()
        try:
            self.project.save()
        except OSError as e:
            self.status.showMessage(f"Could not save project: {e}", 8000)
            return
        self._update_title()
        self.status.showMessage(f"Project saved to {self.project.path}", 5000)

    def _autosave(self) -> None:
        """State changed: schedule a save shortly (does nothing until the
        project has a file; Ctrl+S / Save as… creates it once)."""
        if self.project.path is not None:
            self._autosave_timer.start()

    def _autosave_now(self) -> None:
        try:
            self.project.save()
        except OSError as e:
            self.status.showMessage(f"Could not save project: {e}", 8000)
            return
        self._update_title()
        self.status.showMessage("Autosaved", 3000)

    def closeEvent(self, event) -> None:      # noqa: N802 (Qt naming)
        """Flush any pending autosave before the window closes."""
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()
            self._autosave_now()
        super().closeEvent(event)

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
        self.project.plans_dirs = [
            Path(p) if p else None for p in dialog.plans_dirs()
        ]
        # Re-resolve relative plans dirs against the base; keep guesses for
        # rows the user left empty.
        for i, p in enumerate(self.project.plans_dirs[:]):
            if p is not None and not p.is_absolute():
                self.project.plans_dirs[i] = base / p if (base / p).is_dir() else None
        for i, pd in enumerate(self.project.plans_dirs):
            if pd is None and i < len(dialog.plans_dirs()) \
                    and not dialog.plans_dirs()[i].strip():
                d = self.project.directories[i] if i < len(self.project.directories) else None
                self.project.plans_dirs[i] = guess_plans_dir(d, base) if d else None
        self.project.skip = set(dialog.skip())
        self._autosave()
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
            self._autosave()
        # Move on to the next unconfirmed pair (in the current reading order).
        self.next_unconfirmed()

    def unconfirm_pair(self) -> None:
        keys = self.spread.current_page_keys()
        if len(keys) != 2:
            self.status.showMessage("Only two-page spreads can be unconfirmed", 5000)
            return
        self.project.confirmed.discard(frozenset(keys))
        self.spread.set_confirmed(self.project.confirmed)
        self._autosave()
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
        if key in self.project.skip:
            self.project.skip.discard(key)
            anchor = key          # it comes back; keep it on screen
        else:
            self.project.skip.add(key)
            self.project.roles.pop(key, None)   # skip clears the page's mark
            if partner is not None:
                anchor = partner  # its partner stays put and re-pairs
            else:
                nxt = self.spread._spread_at(self.spread.spread_index() + 1)
                if nxt:
                    anchor = next((self.spread._keys[j] for j in nxt
                                   if j is not None), None)
        self._autosave()          # skip / roles mutated above
        self.reload_book(anchor_key=anchor)

    def mark_page(self, side: int) -> None:
        """Pick a role for one shown page: side 0 = left, 1 = right (the left
        one when the spread shows a single page).

        ``front`` pages get a blank before (solo on the right half, like a
        cover), ``back`` pages get one after (solo on the left half), and
        ``left``/``right`` pair with their neighbors."""
        keys = self.spread.visible_page_keys()
        if not keys:
            return
        key = keys[min(side, len(keys) - 1)]
        role = self.project.roles.get(key)
        menu = QMenu(self)
        choices = [
            ("&Front  (blank before)", ROLE_FRONT),
            ("&Left", ROLE_LEFT),
            ("&Right", ROLE_RIGHT),
            ("Bac&k  (blank after)", ROLE_BACK),
        ]
        actions = []
        for label, r in choices:
            a = menu.addAction(label)
            a.setCheckable(True)
            a.setChecked(role == r)
            actions.append((a, r))
        menu.addSeparator()
        clear = menu.addAction("N&o role")
        clear.setCheckable(True)
        clear.setChecked(role is None)
        chosen = menu.exec(self.cursor().pos())
        if chosen is None:
            return
        for a, r in actions:
            if chosen is a:
                self.project.roles[key] = r
                self.status.showMessage(
                    f"{Path(key).name} marked {r}", 5000
                )
                break
        else:
            self.project.roles.pop(key, None)
        # Keep the marked page on screen: the surrounding spreads shift as
        # the blanks and pairings implied by the role take effect.
        self._autosave()          # role set or cleared above
        self.reload_book(anchor_key=key)

    def rotate_page(self, side: int) -> None:
        """Turn the shown page 90° clockwise per press (wraps through 90, 180,
        270 and back to 0). Rotation is display-only: the files on disk are
        not touched. side 0 = left, 1 = right (the left one when the spread
        shows a single page)."""
        keys = self.spread.visible_page_keys()
        if not keys:
            return
        key = keys[min(side, len(keys) - 1)]
        rot = (self.project.rotations.get(key, 0) + 90) % 360
        if rot == 0:
            self.project.rotations.pop(key, None)   # fully turned back: unmark
        else:
            self.project.rotations[key] = rot
        self._autosave()
        self.reload_book(anchor_key=key)

    def mark_sides(self) -> None:
        """Mark the spread's left page as back and its right page as front:
        the pair splits, the left page keeping its solo-left spread (blank
        after) and the right page its solo-right one (blank before)."""
        keys = self.spread.visible_page_keys()
        if len(keys) != 2:
            self.status.showMessage(
                "Only two-page spreads can be marked left=back / right=front",
                5000,
            )
            return
        left, right = keys
        self.project.roles[left] = ROLE_BACK
        self.project.roles[right] = ROLE_FRONT
        self._autosave()
        self.status.showMessage(
            f"{Path(left).name} marked back, {Path(right).name} marked front",
            5000,
        )
        # Keep the back page on screen (its front partner lands on the
        # adjacent spread).
        self.reload_book(anchor_key=left)

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
        self._autosave()
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

    # ---------- scan sidebars --------------------------------------------

    def _toggle_sidebar(self, side: int) -> None:
        widgets = (self.left_sidebar, self.right_sidebar)
        actions = (self.act_sidebar_left, self.act_sidebar_right)
        widgets[side].setVisible(actions[side].isChecked())

    def _update_sidebars(self) -> None:
        """Fill each sidebar with the contributing scans of the page shown
        on that side (left sidebar = left page, even in reverse mode)."""
        for side, (page, number, entry) in enumerate(self.spread.visible_sides()):
            sidebar = (self.left_sidebar, self.right_sidebar)[side]
            page_plans = getattr(self, "_page_plans", [])
            if page is None and number is None:
                sidebar.set_page("—", "blank", [])
                continue
            index = entry
            title = f"p. {number} — {page.stem}" if page is not None else "blank"
            if page is None:
                sidebar.set_page(title, "blank (before a front / after a back)", [])
                continue
            plans_dir = (page_plans[index]
                         if index is not None and index < len(page_plans) else None)
            info = self.plans.info_for(page, plans_dir)
            if info is None:
                sidebar.set_page(
                    title,
                    f"no plans directory assigned for this book — assign one\n"
                    f"({self.project.directories[0] if self.project.directories else '?'} → "
                    f"Edit project…, Ctrl+E)",
                    [],
                )
                continue
            title += f"\n↳ {info['toml'].parent}" if info["toml"] is not None \
                else ""
            if info["toml"] is None:
                sidebar.set_page(
                    title,
                    f"no plan named {page.stem}.toml in\n{plans_dir}",
                    [],
                )
                continue
            scans = info["scans"]
            msg = info["error"]
            if info["missing"]:
                note = "not found in the plan's scans dir: " + ", ".join(
                    info["missing"]
                )
                msg = (msg + "\n" + note) if msg else note
            if not scans and not msg:
                if info["kind"] == "missing":
                    msg = "plan declares this page missing — no contributing scans"
                else:
                    msg = "no contributing scans listed in the plan"
            sidebar.set_page(title, msg, scans)

    def _toggle_rtl(self) -> None:
        self.spread.set_rtl(self.act_rtl.isChecked())
        self.project.reverse = self.act_rtl.isChecked()
        self._autosave()
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
        self._update_sidebars()
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
        rots = [str(self.project.rotations[k])
                for k in self.spread.current_page_keys()
                if self.project.rotations.get(k)]
        if rots:
            mark += "   ·   ⟳" + "/".join(rots)
        role_names = {
            self.project.roles[k]
            for k in self.spread.current_page_keys()
            if k in self.project.roles
        }
        if role_names:
            mark += "   ·   " + "+".join(sorted(role_names))
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