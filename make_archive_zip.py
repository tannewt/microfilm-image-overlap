#!/usr/bin/env python3
"""Zip the composed microfilm pages for the archive.org book pipeline.

Takes every page PNG in the per-roll directories (unwanted pages are deleted
by hand beforehand, so there is no filtering here), reverses each roll's list
(the rolls were composed back-to-front: each list starts at the film's END
card and finishes at START), renumbers them sequentially, and zips them as
<identifier>_images.zip.

Usage: python make_archive_zip.py [identifier]
"""
import os
import sys

ROLLS = ["P81-4720", "P81-4721", "P81-4722"]  # chronological roll order
PAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")


def main(identifier="galesvilletranscript1894"):
    staging = f"/tmp/{identifier}_staging"
    os.system(f'rm -rf "{staging}"')
    os.makedirs(staging, exist_ok=True)

    kept = 0
    for roll in ROLLS:
        d = os.path.join(PAGES_DIR, roll)
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        files.reverse()  # composed back-to-front; reverse for reading order
        for f in files:
            kept += 1
            os.symlink(os.path.join(d, f),
                       os.path.join(staging, f"{identifier}_{kept:04d}.png"))
        print(f"{roll}: {len(files)} pages (reversed)")
    print(f"total: {kept} pages -> {staging}")

    zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"{identifier}_images.zip")
    rc = os.system(f'cd "{staging}" && zip -0 -X -q "{zip_path}" *.png')
    sys.exit(1 if rc else 0)


if __name__ == "__main__":
    main(*sys.argv[1:])