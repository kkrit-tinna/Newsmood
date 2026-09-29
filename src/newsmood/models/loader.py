"""Fetch the published classifier from the Hub at its pinned revision (T2.4).

snapshot_download mirrors one commit of a Hub repo into cache_dir and returns
the local snapshot folder, which load_classifier() then reads like any
save_pretrained directory. Layout under cache_dir (huggingface_hub 1.32):

    models--<owner>--<name>/
        blobs/<content hash>          the file bytes, stored once
        snapshots/<commit SHA>/<file> symlinks into blobs/
        trees/<commit SHA>.json       that commit's file list, for offline lookups
        refs/<branch>                 branch -> SHA; never written, since we pin a SHA

Never reads training.output_dir (models/local/): that is the training
machine's copy, and scoring from it would skip the published artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import model_info, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from newsmood.config import ModelSettings


def cache_dir(cfg: ModelSettings) -> Path:
    return Path(cfg.cache_dir).expanduser()


def ensure_model(cfg: ModelSettings) -> Path:
    """Local snapshot folder for cfg.repo_id at cfg.revision, downloading it
    on the first call only. The revision is a full commit SHA (config.py
    enforces it), so a cached snapshot can never go stale.

    token=False: the repo is public, and scoring must work, and be tested
    here, the way it does for a user who has no Hub token.
    """
    dest = cache_dir(cfg)
    try:
        return Path(
            snapshot_download(cfg.repo_id, revision=cfg.revision, cache_dir=dest, local_files_only=True, token=False)
        )
    except LocalEntryNotFoundError:
        pass

    info = model_info(cfg.repo_id, revision=cfg.revision, files_metadata=True, token=False)
    size = sum(f.size or 0 for f in info.siblings or [])
    print(
        f"Downloading {cfg.repo_id}@{cfg.revision[:7]} ({size / 1e6:.0f} MB) to {dest}. "
        "One time only; later runs load from this cache.",
        file=sys.stderr,
    )
    return Path(snapshot_download(cfg.repo_id, revision=cfg.revision, cache_dir=dest, token=False))
