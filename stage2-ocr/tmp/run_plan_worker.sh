#!/bin/bash
# Run a page-planning puller inside an incus worker container.
# Usage: run_plan_worker.sh <worker-name> [--workers N]
set -e
cd /root/planwork
PY=/root/ocrwork/.venv-paddle/bin/python
if [ ! -x "$PY" ]; then
  PY=/root/.venv-onnx/bin/python
fi
exec "$PY" /root/planwork/plan_worker_1028.py \
  http://100.104.156.160:8077 "${PLAN_MACHINE:-$(hostname)}:${1:?worker name required}" \
  --workers "${2:-2}"
