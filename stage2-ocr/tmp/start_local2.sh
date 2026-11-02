#!/bin/bash
cd /home/tannewt/repos/image-overlap/stage2-ocr
for port in 1 2 3 4; do
  setsid nohup env OMP_NUM_THREADS=4 OCR_HOST="hoid:local$port" \
    .venv-paddle/bin/python ocr_puller.py http://127.0.0.1:8076 "local$port" \
    > "tmp/puller_local$port.log" 2>&1 < /dev/null &
done
