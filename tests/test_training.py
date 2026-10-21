import json
from pathlib import Path
from scripts import kaggle_job


def test_kaggle_script_contains_shared_code_and_class_list(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_job, "BUILD", tmp_path)
    assert kaggle_job.prepare("test_user") == "test_user/air-canvas-100-classes-v3"
    code = (tmp_path / "train.py").read_text()
    compile(code, "train.py", "exec")
    assert "from core." not in code
    assert 'with_name("classes.json")' not in code
    assert "def prepare_sketch" in code
    assert "class SketchCNN" in code
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
