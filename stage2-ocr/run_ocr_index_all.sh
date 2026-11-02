#!/usr/bin/env bash
set -u
cd /home/tannewt/repos/image-overlap/stage2-ocr
PY=/usr/bin/python3.14
for dir in 20260826-newspaper-scans/P81-4720 20260826-newspaper-scans/P81-4721 20260826-newspaper-scans/P81-4722; do
  echo "=== $(date -Is) START $dir ==="
  "$PY" -u ocr_index.py "$dir"
  echo "=== $(date -Is) DONE  $dir ==="
done
echo "=== $(date -Is) ALL DONE ==="
