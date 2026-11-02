#!/usr/bin/env bash
# Set up the paddle (CPU) OCR environment on a Linux x86_64 machine.
# Run this inside ocr-studio (after copying the folder over).
#
# Matches the versions that work on the master machine:
#   paddlepaddle==3.2.1 (3.3.1/3.0.0 have oneDNN crashes), paddleocr
set -e
cd "$(dirname "$0")"

command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

uv venv .venv-paddle --python 3.13
uv pip install --python .venv-paddle/bin/python -r requirements-ocr.txt

.venv-paddle/bin/python -c "from paddleocr import PaddleOCR; print('paddle env OK')"
echo "models will auto-download (~100 MB) on the first OCR run"