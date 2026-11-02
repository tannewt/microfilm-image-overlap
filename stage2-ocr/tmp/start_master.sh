#!/bin/bash
cd /home/tannewt/repos/image-overlap/stage2-ocr
setsid nohup .venv-paddle/bin/python ./ocr_master_scans.py 8076 > tmp/master_scans.out 2>&1 < /dev/null &
