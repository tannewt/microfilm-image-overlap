#!/bin/bash
# Start the per-edition page-planning master on hoid (port 8077).
# Restart-safe: completed editions are recorded under
# plans/ocr_1028/_meta/<job-id>.json and are skipped.
cd /home/tannewt/repos/image-overlap/stage2-ocr
setsid nohup python plan_master_1028.py 8077 \
  --out-root /home/tannewt/repos/image-overlap/plans/ocr_1028 \
  > tmp/plan_master_1028.log 2>&1 < /dev/null &
