#!/usr/bin/env python3
"""Stage only the CHANGED pages for recomposition, then remove stale ones.

After a re-plan, most page plans are unchanged; recomposing all ~7.5k
full-resolution pages takes ~20 minutes and rewrites ~100 GB.  This helper
compares an old and a new ``plans/ocr_1028`` tree and:

  --prepare (default)
      copy every changed/new plan into a flat-plan tree
      (``<stage>/<roll>/<prefix>/<label>.toml``) so it can be passed to
      ``compose_all_1028.py --plan-root <stage> --force``, and write the
      stale page paths to ``<stage>/stale_pages.txt``;

  --finish
      delete the composed PNGs for plans that no longer exist (stale) and
      remove the staging tree.

Usage:
    python stage2-ocr/apply_plans_1028.py --old /mnt/scratch/plans_old \
        --new plans/ocr_1028 --page-root pages/20261028-newspaper-scans \
        --stage /mnt/scratch/plans_changed_1028 --prepare
    python compose_all_1028.py --plan-root /mnt/scratch/plans_changed_1028 \
        --page-root pages/20261028-newspaper-scans --workers 12 --force
    python stage2-ocr/apply_plans_1028.py ... --finish
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def plan_files(root: Path):
    return {str(p.relative_to(root)): p
            for p in root.glob("*/*/*.toml")}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def page_path(page_root: Path, rel: str) -> Path:
    # rel = <roll>/<prefix>/<label>.toml
    parts = Path(rel).parts
    roll, prefix = parts[0], parts[1]
    label = Path(parts[2]).stem
    return page_root / roll / f"{prefix}_{label}.png"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--page-root", required=True)
    ap.add_argument("--stage", default="/mnt/scratch/plans_changed_1028")
    ap.add_argument("--finish", action="store_true")
    args = ap.parse_args()

    old = plan_files(Path(args.old))
    new = plan_files(Path(args.new))
    page_root = Path(args.page_root)
    stage = Path(args.stage)

    changed = []
    for rel, path in new.items():
        if rel not in old or digest(path) != digest(old[rel]):
            changed.append(rel)
    stale = sorted(set(old) - set(new))

    if args.finish:
        removed = 0
        for rel in stale:
            p = page_path(page_root, rel)
            if p.is_file():
                p.unlink()
                removed += 1
        print(f"Removed {removed} stale page image(s) "
              f"({len(stale)} stale plan(s))")
        shutil.rmtree(stage, ignore_errors=True)
        return 0

    shutil.rmtree(stage, ignore_errors=True)
    for rel in changed:
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(new[rel], dst)
    (stage / "stale_pages.txt").write_text(
        "\n".join(str(page_path(page_root, r)) for r in stale) + "\n")
    stats = {
        "old_plans": len(old),
        "new_plans": len(new),
        "changed_or_new": len(changed),
        "stale": len(stale),
        "unchanged": len(new) - len(changed),
        "stage": str(stage),
    }
    (stage / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print("\nCompose the staged plans with:")
    print(f"  python compose_all_1028.py --plan-root {stage} "
          f"--page-root {page_root} --workers 12 --force")
    print(f"Then finish with --finish to delete the {len(stale)} stale pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
