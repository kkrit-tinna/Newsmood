"""models/loader.py against a fake Hub: snapshot_download and model_info are
monkeypatched, so no test touches the network or the real cache."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from newsmood.config import ModelSettings, Settings
from newsmood.models import loader

SHA = "be7b809e0d8f7bd50e77d902aee199363b7e2a66"


@pytest.fixture
def cfg(tmp_path) -> ModelSettings:
    return ModelSettings(repo_id="owner/model", revision=SHA, cache_dir=str(tmp_path / "cache"))


class FakeHub:
    """Records every call. The cache holds the snapshot once a non-offline
    snapshot_download has run, like the real one."""

    def __init__(self, monkeypatch, cached: bool):
        self.cached = cached
        self.snapshot_calls: list[dict] = []
        self.info_calls: list[dict] = []
        monkeypatch.setattr(loader, "snapshot_download", self.snapshot_download)
        monkeypatch.setattr(loader, "model_info", self.model_info)

    def snapshot_download(self, repo_id, **kwargs):
        self.snapshot_calls.append({"repo_id": repo_id, **kwargs})
        if kwargs.get("local_files_only") and not self.cached:
            raise LocalEntryNotFoundError("not cached")
        self.cached = True
        return str(Path(kwargs["cache_dir"]) / "snapshots" / kwargs["revision"])

    def model_info(self, repo_id, **kwargs):
        self.info_calls.append({"repo_id": repo_id, **kwargs})
        return SimpleNamespace(siblings=[SimpleNamespace(size=265_000_000), SimpleNamespace(size=700_000)])


def test_miss_downloads_the_pinned_sha_into_the_newsmood_cache_and_prints_one_notice(cfg, monkeypatch, capsys):
    hub = FakeHub(monkeypatch, cached=False)

    path = loader.ensure_model(cfg)

    assert [c.get("local_files_only", False) for c in hub.snapshot_calls] == [True, False]
    for call in hub.snapshot_calls + hub.info_calls:
        assert call["repo_id"] == "owner/model"
        assert call["revision"] == SHA
    for call in hub.snapshot_calls:
        assert call["cache_dir"] == Path(cfg.cache_dir)
        assert "local_dir" not in call
    assert hub.info_calls[0]["files_metadata"] is True
    assert path == Path(cfg.cache_dir) / "snapshots" / SHA

    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    assert "266 MB" in err  # sum of every file's size
    assert cfg.cache_dir in err
    assert "One time only" in err


def test_hit_loads_from_cache_without_network_or_notice(cfg, monkeypatch, capsys):
    hub = FakeHub(monkeypatch, cached=True)

    loader.ensure_model(cfg)

    assert len(hub.snapshot_calls) == 1
    assert hub.snapshot_calls[0]["local_files_only"] is True
    assert hub.snapshot_calls[0]["revision"] == SHA
    assert hub.info_calls == []
    assert capsys.readouterr().err == ""


def test_second_call_is_a_silent_hit(cfg, monkeypatch, capsys):
    hub = FakeHub(monkeypatch, cached=False)
    loader.ensure_model(cfg)
    capsys.readouterr()

    loader.ensure_model(cfg)

    assert capsys.readouterr().err == ""
    assert len(hub.info_calls) == 1


def test_never_passes_main_or_a_moving_ref(cfg, monkeypatch):
    hub = FakeHub(monkeypatch, cached=False)
    loader.ensure_model(cfg)
    assert {c["revision"] for c in hub.snapshot_calls + hub.info_calls} == {SHA}


def test_downloads_anonymously(cfg, monkeypatch):
    # Public repo: scoring must not depend on whatever token the machine has.
    hub = FakeHub(monkeypatch, cached=False)
    loader.ensure_model(cfg)
    assert all(c["token"] is False for c in hub.snapshot_calls + hub.info_calls)


def test_real_config_caches_under_home_dot_cache_newsmood_not_models_local():
    settings = Settings()
    assert loader.cache_dir(settings.model) == Path.home() / ".cache" / "newsmood"
    assert Path(settings.training.output_dir).resolve() != loader.cache_dir(settings.model).resolve()
