"""Precedence: explicit kwargs > env > config/default.yaml > code default.

Settings.settings_customise_sources ranks sources by position (earlier
wins) — this asserts that ordering directly, rather than relying on some
later task's settings-driven path resolution to notice a regression by
accident. See IMPLEMENTATION_GUIDE.md §9 Deviations, 2026-09-18.
"""

from __future__ import annotations

from pathlib import Path

import yaml as yamllib
from pydantic_settings import BaseSettings, SettingsConfigDict

import newsmood.config as config_module
from newsmood.config import DatasetSettings, Settings, VaderSettings, YamlConfigSource

_AGREEMENT_CONFIGS = [
    "sentences_allagree",
    "sentences_75agree",
    "sentences_66agree",
    "sentences_50agree",
]


def _dataset_kwargs(**overrides) -> DatasetSettings:
    base = dict(
        repo_id="kwarg-repo",
        phrasebank_config="sentences_75agree",
        agreement_configs=_AGREEMENT_CONFIGS,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=1,
        tokenizer="t",
        splits_dir="kwarg-splits",
        sample_size=1,
        sample_path="kwarg-sample",
    )
    base.update(overrides)
    return DatasetSettings(**base)


def _write_yaml_config(path: Path) -> None:
    path.write_text(
        yamllib.safe_dump(
            {
                "dataset": {
                    "repo_id": "yaml-repo",
                    "phrasebank_config": "sentences_75agree",
                    "agreement_configs": _AGREEMENT_CONFIGS,
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                    "seed": 2,
                    "tokenizer": "t",
                    "splits_dir": "yaml-splits",
                    "sample_size": 1,
                    "sample_path": "yaml-sample",
                },
                "vader": {"positive_threshold": 0.22, "negative_threshold": -0.22},
            }
        )
    )


def _point_settings_at_yaml(tmp_path: Path, monkeypatch) -> None:
    """Redirect Settings' yaml source at a throwaway file with known
    sentinel values, so these tests don't depend on the real
    config/default.yaml and aren't broken by future edits to it."""
    yaml_path = tmp_path / "config.yaml"
    _write_yaml_config(yaml_path)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", yaml_path)


# --- yaml > (nothing else set) ---


def test_yaml_wins_when_no_env_or_kwargs(tmp_path, monkeypatch):
    _point_settings_at_yaml(tmp_path, monkeypatch)
    monkeypatch.delenv("NEWSMOOD_DATASET__SPLITS_DIR", raising=False)

    settings = Settings()

    assert settings.dataset.splits_dir == "yaml-splits"
    assert settings.vader.positive_threshold == 0.22


# --- env > yaml ---


def test_env_beats_yaml(tmp_path, monkeypatch):
    _point_settings_at_yaml(tmp_path, monkeypatch)
    monkeypatch.setenv("NEWSMOOD_DATASET__SPLITS_DIR", "env-splits")

    settings = Settings()

    assert settings.dataset.splits_dir == "env-splits"
    # a field the env var didn't touch still falls through to yaml
    assert settings.vader.positive_threshold == 0.22


# --- explicit kwargs > env ---


def test_kwargs_beat_env(tmp_path, monkeypatch):
    _point_settings_at_yaml(tmp_path, monkeypatch)
    monkeypatch.setenv("NEWSMOOD_DATASET__SPLITS_DIR", "env-splits")

    settings = Settings(dataset=_dataset_kwargs(splits_dir="kwarg-splits"))

    assert settings.dataset.splits_dir == "kwarg-splits"


# --- explicit kwargs > env > yaml, chained ---


def test_kwargs_beat_env_which_beats_yaml_in_one_settings_object(tmp_path, monkeypatch):
    _point_settings_at_yaml(tmp_path, monkeypatch)
    monkeypatch.setenv("NEWSMOOD_DATASET__SPLITS_DIR", "env-splits")

    settings = Settings(dataset=_dataset_kwargs(splits_dir="kwarg-splits"))

    # this field was overridden by kwargs
    assert settings.dataset.splits_dir == "kwarg-splits"
    # this field was never touched by kwargs or env, so it still falls
    # through past both to yaml
    assert settings.vader.positive_threshold == 0.22


# --- yaml > code (field) default ---
#
# DatasetSettings and VaderSettings declare every field required, so there
# is no field-default tier to observe on the real Settings object. This
# probes the shared mechanism (YamlConfigSource + the same source order)
# directly with a model that has one, instead of adding an unused default
# field to production settings just to exercise this tier.


def _probe_settings_class(yaml_path: Path) -> type[BaseSettings]:
    class _Probe(BaseSettings):
        model_config = SettingsConfigDict(env_prefix="NEWSMOOD_PROBE_")

        value: str = "field-default"

        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                env_settings,
                YamlConfigSource(settings_cls, yaml_path),
                dotenv_settings,
                file_secret_settings,
            )

    return _Probe


def test_field_default_used_when_yaml_omits_the_key(tmp_path, monkeypatch):
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text("{}\n")
    monkeypatch.delenv("NEWSMOOD_PROBE_VALUE", raising=False)

    probe_cls = _probe_settings_class(yaml_path)

    assert probe_cls().value == "field-default"


def test_yaml_beats_field_default(tmp_path, monkeypatch):
    yaml_path = tmp_path / "with_value.yaml"
    yaml_path.write_text("value: from-yaml\n")
    monkeypatch.delenv("NEWSMOOD_PROBE_VALUE", raising=False)

    probe_cls = _probe_settings_class(yaml_path)

    assert probe_cls().value == "from-yaml"
