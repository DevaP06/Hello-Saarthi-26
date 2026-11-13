from __future__ import annotations

import pytest

from src.knowledge_base.config import KBConfig, load_config


def test_load_config_raises_clear_error_without_data_root(tmp_path, monkeypatch):
    monkeypatch.delenv("SAARTHI_KB_DATA_ROOT", raising=False)
    missing_yaml = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ValueError, match="SAARTHI_KB_DATA_ROOT"):
        load_config(missing_yaml)


def test_env_var_overrides_yaml_value(tmp_path, monkeypatch):
    yaml_path = tmp_path / "kb.yaml"
    yaml_path.write_text(f"data_root: '{(tmp_path / 'from_yaml').as_posix()}'\nembedding_batch_size: 8\n", encoding="utf-8")
    monkeypatch.setenv("SAARTHI_KB_EMBED_BATCH_SIZE", "32")

    config = load_config(yaml_path)

    assert config.embedding_batch_size == 32  # env var wins over YAML
    assert str(config.data_root) == str(tmp_path / "from_yaml")  # YAML value kept when no env override


def test_data_root_env_var_overrides_yaml_path(tmp_path, monkeypatch):
    yaml_path = tmp_path / "kb.yaml"
    yaml_path.write_text(f"data_root: '{(tmp_path / 'from_yaml').as_posix()}'\n", encoding="utf-8")
    monkeypatch.setenv("SAARTHI_KB_DATA_ROOT", str(tmp_path / "from_env"))

    config = load_config(yaml_path)

    assert str(config.data_root) == str(tmp_path / "from_env")


def test_ensure_directories_creates_every_subdirectory(tmp_path):
    config = KBConfig(data_root=tmp_path / "root")
    config.ensure_directories()

    assert config.normalized_dir.exists()
    assert config.chunks_dir.exists()
    assert config.embeddings_dir.exists()
    assert config.manifests_dir.exists()
    assert config.reports_dir.exists()
