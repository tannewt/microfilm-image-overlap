#!/bin/bash
# Auto-stop the 1028 OCR run 20 minutes from launch (laptop sleep).
cd /home/tannewt/repos/image-overlap/stage2-ocr
sleep 1200
{
  echo "=== autostop $(date '+%F %T')"
  curl -s -m 5 http://127.0.0.1:8076/stats; echo
} >> tmp/autostop_1028.log 2>&1
for c in dalinar:gpu1 dalinar:ocr1 clanker:ocr1 clanker:ocr2 clanker:ocr-arch; do
  incus exec "$c" --project sandboxes -- bash -c 'systemctl stop "ocr-pull*" "gpu-ocr*" 2>/dev/null' >> tmp/autostop_1028.log 2>&1
done
for c in dalinar:gpu1 dalinar:ocr1 clanker:ocr1 clanker:ocr2 clanker:ocr-arch; do
  incus stop "$c" --project sandboxes >> tmp/autostop_1028.log 2>&1
done
for p in $(pgrep -f '[o]cr_puller.py http://127.0.0.1:8076'); do kill "$p"; done
for p in $(pgrep -f '[o]cr_master_scans.py 8076'); do kill "$p"; done
echo "stopped $(date '+%F %T')" >> tmp/autostop_1028.log
