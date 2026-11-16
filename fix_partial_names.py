#!/usr/bin/env python3
"""Repair scan filenames whose leading batch prefix was lost or truncated.

The scanner names scans ``<prefix>_<settings>_<index>.tif``.  Occasionally a
file lands with only part of the prefix, e.g.::

    275_6200_m0.240_f0.8_d0.2_s1.3_0009.tif
    322_6200_m0.240_f0.8_d0.2_s1.3_0017.tif
    _6200_m0.240_f0.8_d0.2_s1.3_0057.tif

while the rest of the same batch is named ``00275_..._0009.tif``,
``05322_..._0017.tif``, etc.  The trailing index survives, so a partial name
can be matched to the complete batch that has a gap at that index and whose
stem ends with the partial stem.  Unambiguous matches are renamed; anything
else is reported and left alone.

Dry-run by default; pass ``--apply`` to actually rename.

Usage:
    python fix_partial_names.py DIRECTORY
    python fix_partial_names.py DIRECTORY --apply
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Everything before the trailing _NNNN plus extension.  Greedy .* is
# deliberate: the stem contains underscores and dots of its own.
NAME_RE = re.compile(r"^(?P<stem>.*)_(?P<index>\d+)(?P<ext>\.[^.]+)$")

GREEN = "\033[32m"
WHITE = "\033[37m"
RESET = "\033[0m"


def scan(directory: Path, index_width: int):
    """Return {stem: {index: [paths]}} for names with a trailing index."""
    groups: dict[str, dict[int, list[Path]]] = {}
    ignored: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        m = NAME_RE.match(path.name)
        if not m or len(m.group("index")) != index_width:
            ignored.append(path)
            continue
        stem = m.group("stem")
        index = int(m.group("index"))
        groups.setdefault(stem, {}).setdefault(index, []).append(path)
    return groups, ignored


def suffix_relations(stems: list[str]):
    """Split stems into complete stems and partial stems.

    A stem is partial when a *longer* stem ends with it.  With the scanner
    naming, ``275_...`` is a suffix of ``00275_...`` and ``_...`` is a suffix
    of every stem.  A complete stem has no longer stem ending with it.
    """
    partial: dict[str, list[str]] = {}
    for stem in stems:
        supers = [other for other in stems
                  if len(other) > len(stem) and other.endswith(stem)]
        if supers:
            partial[stem] = supers
    return partial


def plan_renames(groups, partial, complete):
    """Return (renames, unresolved), where renames is [(src, dst)]."""
    renames: list[tuple[Path, Path]] = []
    unresolved: list[tuple[Path, str]] = []
    planned_dsts: set[Path] = set()

    for stem, indices in sorted(groups.items()):
        if stem not in partial:
            continue
        for index, paths in sorted(indices.items()):
            candidates = []
            for full in partial[stem]:
                if full not in complete:
                    continue
                full_indices = groups[full]
                lo, hi = min(full_indices), max(full_indices)
                if lo <= index <= hi and index not in full_indices:
                    candidates.append(full)

            if len(candidates) != 1:
                reason = ("no complete batch has a gap at this index"
                          if not candidates else
                          "ambiguous: " + ", ".join(candidates))
                unresolved.extend((p, reason) for p in paths)
                continue

            full = candidates[0]
            for src in paths:
                # Preserve the original zero padding exactly.
                digits = NAME_RE.match(src.name).group("index")
                dst = src.with_name(f"{full}_{digits}{src.suffix}")
                if dst in planned_dsts:
                    unresolved.append((src, f"another file already maps to: "
                                            f"{dst.name}"))
                elif dst.exists() and dst != src:
                    unresolved.append((src, f"destination exists: {dst.name}"))
                else:
                    planned_dsts.add(dst)
                    renames.append((src, dst))
    return renames, unresolved


def common_suffix_len(a: str, b: str) -> int:
    """Length of the common suffix of two strings."""
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def format_destination(src: Path, dst: Path, color: bool) -> str:
    """The complete destination name, with the reconstructed part in green."""
    if not color:
        return dst.name
    common = common_suffix_len(src.name, dst.name)
    added = dst.name[:len(dst.name) - common]
    rest = dst.name[len(added):]
    return f"{GREEN}{added}{WHITE}{rest}{RESET}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path,
                        help="batch folder to inspect")
    parser.add_argument("--apply", action="store_true",
                        help="perform the renames (default: dry run)")
    parser.add_argument("--index-width", type=int, default=4,
                        help="number of trailing index digits (default: 4)")
    parser.add_argument("--color", choices=("auto", "always", "never"),
                        default="auto",
                        help="color the reconstructed filename prefix "
                             "(default: auto)")
    args = parser.parse_args(argv)

    color = (args.color == "always" or
             (args.color == "auto" and sys.stdout.isatty()
              and "NO_COLOR" not in os.environ))

    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")

    groups, ignored = scan(args.directory, args.index_width)
    if not groups:
        print(f"no <prefix>_<{args.index_width}-digit index> filenames in "
              f"{args.directory}")
        return 1

    partial = suffix_relations(list(groups))
    complete = [s for s in groups if s not in partial]

    print(f"directory: {args.directory}")
    print(f"complete batches: {len(complete)}")
    for stem in sorted(complete, key=lambda s: min(groups[s])):
        indices = groups[stem]
        print(f"  {stem}: {len(indices)} files, "
              f"{min(indices):0{args.index_width}d}-"
              f"{max(indices):0{args.index_width}d}")

    if partial:
        print(f"partial batches: {len(partial)}")
        for stem in sorted(partial, key=lambda s: min(groups[s])):
            print(f"  {stem!r}: {len(groups[stem])} file(s)")

    renames, unresolved = plan_renames(groups, partial, set(complete))
    if renames:
        print(f"\n{'rename' if args.apply else 'would rename'} "
              f"{len(renames)} file(s):")
        for src, dst in renames:
            print(f"  {format_destination(src, dst, color)}")
            if args.apply:
                os.rename(src, dst)
    else:
        print("\nno unambiguous renames found")

    if unresolved:
        print(f"\nleft alone ({len(unresolved)}):")
        for path, reason in unresolved:
            print(f"  {path.name}: {reason}")
    if ignored:
        print(f"\nignored (no trailing {args.index_width}-digit index):")
        for path in ignored:
            print(f"  {path.name}")

    if renames and not args.apply:
        print("\ndry run; pass --apply to rename")
    return 0


if __name__ == "__main__":
    sys.exit(main())
