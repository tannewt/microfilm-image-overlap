#!/usr/bin/env python3
"""
Parse a sequence_overlap.py text run and write out the anomalous pairs:
  - NO OVERLAP pairs (gaps / no shared content)
  - NEAR-DUPLICATE pairs (>50% area overlap with a small vertical shift,
    i.e. the film barely advanced between the two frames)

Usage:
    python extract_anomalies.py RUN_OUTPUT.txt [-o OUT.txt] [--near-dup AREA]
"""
import argparse
import re
import sys

OK_RE = re.compile(
    r'^(.*?\.tif) -> (.*?\.tif):\s+([\d.]+)% area,\s+([\d.]+)% vert,\s+'
    r'dx=([+-]?\d+)\s+dy=([+-]?\d+),\s+(\d+)\s+matches,\s+IQR\(dy\)=([\d.]+)'
)
NO_RE = re.compile(
    r'^(.*?\.tif) -> (.*?\.tif):\s+NO OVERLAP \((\d+)\s+matches,\s+'
    r'IQR\(dy\)=([\d.]+)px'
)


def parse(path):
    rows = []
    for ln in open(path):
        ln = ln.rstrip("\n")
        m = OK_RE.match(ln)
        if m:
            rows.append(dict(a=m.group(1), b=m.group(2), status="ok",
                             area=float(m.group(3)), vert=float(m.group(4)),
                             dx=int(m.group(5)), dy=int(m.group(6)),
                             matches=int(m.group(7)), iqr=float(m.group(8))))
            continue
        m = NO_RE.match(ln)
        if m:
            rows.append(dict(a=m.group(1), b=m.group(2), status="no_overlap",
                             area=0.0, vert=0.0, dx=0, dy=0,
                             matches=int(m.group(3)), iqr=float(m.group(4))))
    return rows


def fmt(r):
    if r["status"] != "ok":
        return ("  {a} -> {b}\n"
                "      NO OVERLAP  matches={matches}  IQR(dy)={iqr:.0f}px"
                .format(**r))
    return ("  {a} -> {b}\n"
            "      {area:5.1f}% area, {vert:5.1f}% vert, dy={dy:+d}, "
            "matches={matches}, IQR(dy)={iqr:.0f}".format(**r))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_output", help="sequence_overlap.py stdout text file")
    ap.add_argument("-o", "--out", default=None, help="output file (default: stdout)")
    ap.add_argument("--near-dup", type=float, default=50.0,
                    help="area %% above which a pair counts as near-duplicate "
                         "(default: 50.0)")
    args = ap.parse_args()

    rows = parse(args.run_output)
    no_overlap = [r for r in rows if r["status"] != "ok"]
    near_dup = [r for r in rows if r["status"] == "ok" and r["area"] > args.near_dup]
    normal = [r for r in rows if r["status"] == "ok" and r["area"] <= args.near_dup]

    out = []
    out.append("Overlap anomalies")
    out.append("=" * 60)
    out.append("Total pairs: {}".format(len(rows)))
    out.append("  normal (<={:.0f}% area): {}".format(args.near_dup, len(normal)))
    out.append("  no overlap:           {}".format(len(no_overlap)))
    out.append("  near-duplicate (>{:.0f}% area, small shift): {}".format(
        args.near_dup, len(near_dup)))
    out.append("")

    out.append("NEAR-DUPLICATE pairs (>{:.0f}% area, small vertical shift)".format(
        args.near_dup))
    out.append("-" * 60)
    for r in sorted(near_dup, key=lambda r: -r["area"]):
        out.append(fmt(r))
    out.append("")

    out.append("NO-OVERLAP pairs")
    out.append("-" * 60)
    for r in no_overlap:
        out.append(fmt(r))

    text = "\n".join(out) + "\n"
    if args.out:
        open(args.out, "w").write(text)
        print("wrote {} ({} no-overlap, {} near-duplicate)".format(
            args.out, len(no_overlap), len(near_dup)))
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()