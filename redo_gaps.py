#!/usr/bin/env python3
"""Re-plan + re-composite the pages around each gap junction without redoing
the whole roll.

For every known gap, run plan_pages on a small --only subset spanning the
junction (±2 scans, so the red-band check sees real neighbors), then
compose_pages renders the subset's plans and the joined page / missing-file
red pages are copied into the roll's pages/ directory and the stale split
pieces dropped. Interior pages recompute identically (page assembly is
dx/offset-relative), so only the junction artifacts need copying.
"""
import glob
import os
import re
import shutil
import subprocess
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
WORK = '/tmp/gapfix'

ROLLS = [
    dict(roll='P81-4720', prefix='06818-6300_0.2720',
         joins=[(5, 6), (189, 190), (203, 204), (248, 249), (262, 263),
                (293, 294), (305, 306), (314, 315), (326, 327), (340, 341),
                (350, 351), (403, 404),
                (744, 745), (745, 746), (746, 747), (747, 748), (748, 749),
                (749, 750), (750, 751), (751, 752), (752, 753), (753, 754),
                (754, 755), (755, 756), (756, 757),
                (760, 761), (761, 762), (762, 763), (763, 764),
                (772, 773), (777, 778), (779, 780), (784, 785)],
         missing=[46, 132, 218],
         subsets=[(3, 8), (44, 48), (130, 134), (187, 192), (201, 206),
                  (215, 221), (246, 251), (260, 265), (291, 296), (303, 308),
                  (312, 317), (324, 329), (338, 343), (348, 353), (401, 406),
                  (742, 787)]),
    dict(roll='P81-4721', prefix='04860-6300_0.2720',
         joins=[(724, 725), (832, 833)],
         missing=[126],
         subsets=[(123, 129), (722, 727), (830, 833)]),
    dict(roll='P81-4722', prefix='07278-6300_0.2720',
         joins=[(3, 4), (121, 122), (179, 180), (202, 203), (217, 218),
                (236, 237), (262, 263), (321, 322), (335, 336), (376, 377),
                (394, 395), (495, 496), (506, 507), (603, 604), (704, 705),
                (935, 936)],
         missing=[],
         subsets=[(1, 6), (119, 124), (177, 182), (200, 205), (215, 220),
                  (234, 239), (260, 265), (319, 324), (333, 338), (374, 379),
                  (392, 397), (493, 498), (504, 509), (601, 606), (702, 707),
                  (933, 936)]),
]


def pair_status(log_text, prefix, a, b):
    """'no_overlap' | 'missing_frame' | 'ok' | 'absent' for one pair line."""
    m = re.search(rf'_{a:04d}\.tif -> {prefix}_{b:04d}\.tif:(.*)',
                  log_text)
    if not m:
        return 'absent'
    tail = m.group(1)
    if 'no_overlap' in tail:
        return 'no_overlap'
    if 'missing_frame' in tail:
        return 'missing_frame'
    return 'ok'


def main():
    for spec in ROLLS:
        roll, prefix = spec['roll'], spec['prefix']
        scans = f'20260826-newspaper-scans/{roll}'
        pages = f'pages/{roll}'
        print(f'===== {roll}')
        for lo, hi in spec['subsets']:
            tag = f'{lo:04d}-{hi:04d}'
            outdir = os.path.join(WORK, roll, tag)
            pagedir = os.path.join(outdir, 'pages')
            os.makedirs(outdir, exist_ok=True)
            only = ','.join(
                f'{prefix}_{i:04d}.tif'
                for i in range(lo, hi + 1)
                if os.path.exists(os.path.join(scans, f'{prefix}_{i:04d}.tif')))
            log = os.path.join(WORK, roll, f'{tag}.log')
            with open(log, 'w') as f:
                rc = subprocess.call(
                    [sys.executable, 'plan_pages.py', scans,
                     '--pattern', f'^{prefix}', '--cache-dir', 'sift_cache',
                     '--output-dir', outdir, '--only', only],
                    stdout=f, stderr=subprocess.STDOUT)
            if rc != 0:
                print(f'  subset {tag}: plan_pages FAILED rc={rc} ({log})')
                continue
            with open(log, 'a') as f:
                rc = subprocess.call(
                    [sys.executable, 'compose_pages.py', outdir,
                     '--scan-dir', scans, '--output-dir', pagedir],
                    stdout=f, stderr=subprocess.STDOUT)
            if rc != 0:
                print(f'  subset {tag}: compose_pages FAILED rc={rc} ({log})')
                continue
            text = open(log, errors='replace').read()

            def g1(stem):
                return sorted(glob.glob(os.path.join(pagedir, stem + '.*')))

            # missing-file red pages inside this subset
            for seq in spec['missing']:
                if lo <= seq <= hi:
                    made = g1(f'scans{seq:03d}-{seq:03d}*')
                    if made:
                        shutil.copy2(made[0], pages)
                        print(f'  {tag}: red page for missing {seq:04d} copied')
                    else:
                        print(f'  {tag}: WARNING no red page for {seq:04d}')

            # junctions inside this subset
            for a, b in spec['joins']:
                if not (lo <= a and b <= hi):
                    continue
                status = pair_status(text, prefix, a, b)
                if status == 'ok':
                    print(f'  {tag}: junction {a:04d}->{b:04d} now LINKED '
                          f'(no action)')
                    continue
                if status == 'missing_frame':
                    print(f'  {tag}: junction {a:04d}->{b:04d} missing_frame '
                          f'(red page only; pieces kept)')
                    continue
                joined = g1(f'scans{a:03d}-{b:03d}-gap*')
                if joined:
                    for stale in (glob.glob(os.path.join(
                            pages, f'scans{a:03d}-{a:03d}*.*')) +
                            glob.glob(os.path.join(
                            pages, f'scans{b:03d}-{b:03d}*.*'))):
                        os.remove(stale)
                    for j in joined:
                        shutil.copy2(j, pages)
                    print(f'  {tag}: junction {a:04d}->{b:04d} joined -> '
                          f'{pages} (stale pieces removed)')
                else:
                    print(f'  {tag}: junction {a:04d}->{b:04d} still a gap '
                          f'but not joined (pieces left as-is) -- check log')


if __name__ == '__main__':
    main()