"""Shared pytest fixtures and path setup for the image-overlap test suite.

The repo's modules (check_overlap_sift.py, sequence_overlap.py,
batch_report.py, ...) live at the repo root, so this conftest puts the repo
root on sys.path. Fixtures here skip tests whose data isn't present, so the
suite runs wherever the scan data + SIFT cache happen to be (and skips
gracefully in CI without them).
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)


@pytest.fixture(scope="session")
def data_dir():
    """Directory holding the real scan batches. Skips the test if absent."""
    d = os.path.join(REPO, "scans2", "P81-4720")
    if not os.path.isdir(d):
        pytest.skip(f"scan data not present: {d}")
    return d


@pytest.fixture(scope="session")
def cache_dir():
    """Shared on-disk SIFT cache dir. Skips the test if absent."""
    d = os.path.join(REPO, "sift_cache")
    if not os.path.isdir(d):
        pytest.skip(f"SIFT cache not present: {d}")
    return d


def pair(data_dir, prefix, idx):
    """Absolute path to one scan, given its batch prefix and 4-digit index."""
    return os.path.join(data_dir, f"{prefix}_{idx:04d}.tif")