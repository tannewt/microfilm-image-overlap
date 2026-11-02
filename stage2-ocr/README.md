# stage2-ocr — OCR over the composed pages and raw scans

Stage 2 of the microfilm pipeline: everything that turns page images into
text. Stage 1 (`stage1-circuitpython/`, the scanner firmware) captures and
overlaps the scans; the top level of this repo (`plan_pages.py`,
`compose_pages.py`) composes them into one image per microfilm page. This
folder reads those page/scan images and produces line-level OCR JSON.

This used to be a separate `ocr-studio` repo. It now lives here; its history
was merged in and every path below is relative to *this* folder, so run the
commands from `stage2-ocr/` (or use the absolute paths in the launcher
scripts under `tmp/`).

## Entry points

- **`ocr_worker_server.py`** — the worker: an HTTP server that takes a page
  image and returns OCR JSON. Runs on CPU or GPU.
- **`ocr_worker_onnx.py` / `ocr_worker_ncnn.py`** — standalone worker
  implementations (onnxruntime-MIGraphX, and ncnn/Vulkan) behind the same
  protocol; `ocr_worker_server.py` is the fleet-facing wrapper.
- **`ocr_puller*.py`** — pullers that claim jobs off a master and push
  results back (local, GPU, GPU2, ncnn flavors — see `WORKERS.md`).
- **`ocr_master_1006.py` / `ocr_master_scans.py`** — the masters that hand
  out work and collect `ocr_results/`.
- **`plan_master_1028.py` / `plan_worker_1028.py`** — planning pass that maps
  scans to jobs for the 20261028 batch.
- **`dispatch_ocr.py`** — worker-as-a-service dispatcher (see `DISTRIBUTE.md`).
- **`ocr_eval.py`** — score engine output against the golden transcriptions
  (`golden/`); numbers live in `BENCHMARKS.md`.
- **`catalog_front_pages_1028.py`** — catalog front pages / volume markers
  into `front_pages_1028.json` (+ `FRONT_PAGES_1028.md`).
- **`page_structure.py`** / **`clip_article.py`** — classify newspaper
  structure and cut self-contained article clips (`clips/`, `CLIPS_INDEX.md`).
- **`ocr_index.py` / `ocr_grep.py`** — index and search the OCR output.
- **`scripts/check_overlap_offsets.py`** — validate PP line-box geometry
  against the page plans' film offsets in this repo's `plans/`.

## Docs

- `WORKERS.md` — rolling out the worker fleet.
- `SCAN_OCR_RUN.md` — the big distributed scan-level corpus run.
- `DISTRIBUTE.md` — distributing a run over several machines.
- `INCUS-FLEET.md` — reaching the remote machines.
- `GLM-OCR.md` — GLM-OCR integration notes; `gpuwork/README.md` — GPU worker deploy.
- `BENCHMARKS.md` — engine comparison against the golden set.
- `STRUCTURE-AND-LINKING.md` — structure/entity design.
- Surname studies from the OCR output: `PARKERS.md`, `TRUAXES.md`,
  `NELSONS.md`, `MCWAIN.md`, `VERMONT.md`, plus the name-search reports
  `NAME-SEARCH-1006.md`, `NAME-SEARCH-SUMMARY.md`, `SEATTLE-NAME-SEARCH.md`.

## Data (gitignored)

`ocr_results/`, `clips/`, `shards/`, `bench_pages/`, `P81-4720..4722/`,
`20260826-newspaper-scans/`, `tmp/`, and the two local venvs stay out of git;
rebuild them with the scripts above (`scripts/make_bench_pages.py`,
`setup_paddle_env.sh`, `requirements-ocr.txt`, `requirements-onnx.txt`).
Model weights under `onnx_models/`, the rec charset under `models/`, and the
golden transcriptions under `golden/` *are* tracked.

The venvs were moved with the folder: their `bin/` scripts and `activate`
files were repointed at the new path, but they were built against a temp
CPython under `/tmp/uvpy` that no longer exists, so recreate them if they
misbehave.
