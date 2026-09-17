from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


class DatasetSettings(BaseModel):
    repo_id: str
    phrasebank_config: str
    agreement_configs: list[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    tokenizer: str
    splits_dir: str
    sample_size: int
    sample_path: str


class YamlConfigSource(PydanticBaseSettingsSource):
    """Reads config/default.yaml as a settings layer, below env vars."""

    def __init__(self, settings_cls: type[BaseSettings], yaml_file: Path):
        super().__init__(settings_cls)
        self.yaml_file = yaml_file

    def get_field_value(self, field, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        with open(self.yaml_file) as f:
            return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    """Precedence env > config/default.yaml > code default, per CLAUDE.md."""

    model_config = SettingsConfigDict(env_prefix="NEWSMOOD_", env_nested_delimiter="__")

    dataset: DatasetSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            YamlConfigSource(settings_cls, DEFAULT_CONFIG_PATH),
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )


def get_settings() -> Settings:
    return Settings()
