import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_e2_dataset.py"


def load_script():
    assert SCRIPT.is_file(), "The released E2 dataset needs a machine-local trainer entry point"
    spec = importlib.util.spec_from_file_location("prepare_e2_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_release(tmp_path, *, omit_label=False):
    archive = tmp_path / "release.zip"
    prefix = "formal_photos/yolo_export/"
    yaml = (
        "path: /another/training/machine\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnc: 6\nnames: ['bottle', 'cup', 'keyboard', 'mouse', "
        "'book', 'scissors']\n"
    )
    splits = {"train": ["G06P1"], "val": ["G01P1"], "test": ["G02P1"]}
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(
            "formal_photos/splits.json", json.dumps({"dataset_release": "v1", "splits": splits})
        )
        z.writestr(prefix + "dataset.yaml", yaml)
        for split, (pair,) in splits.items():
            for side in ["before", "after"]:
                name = f"{pair}__{side}"
                z.writestr(f"{prefix}images/{split}/{name}.jpg", f"image-{name}")
                if not (omit_label and split == "val" and side == "after"):
                    z.writestr(f"{prefix}labels/{split}/{name}.txt", "0 .5 .5 .1 .1\n")
    return archive


def test_relocate_hashes_and_excludes_test(tmp_path, monkeypatch):
    mod = load_script()
    archive = make_release(tmp_path)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(mod, "FORMAL_V1_SHA256", sha)
    target = tmp_path / "fresh directory"
    mod.prepare_dataset(archive, target)
    yaml = (target / "dataset.yaml").read_text()
    assert f"path: {target.resolve().as_posix()}\n" in yaml
    assert "test:" not in yaml and "/another/training/machine" not in yaml
    assert not (target / "images/test").exists() and not (target / "labels/test").exists()
    receipt = json.loads((target / "source_receipt.json").read_bytes())
    assert receipt["archive_sha256"] == sha and receipt["images_per_split"] == {
        "train": 2,
        "val": 2,
    }
    assert (
        receipt["local_yaml_sha256"]
        == hashlib.sha256((target / "dataset.yaml").read_bytes()).hexdigest()
    )
    assert receipt["source_yaml_sha256"] != receipt["local_yaml_sha256"]
    assert len(receipt["member_sha256"]) == 8
    for name, digest in receipt["member_sha256"].items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        mod.prepare_dataset(archive, target)
    assert (target / "images/train/G06P1__before.jpg").read_bytes() == b"image-G06P1__before"


@pytest.mark.parametrize("name", ["dataset # archive", "dataset:"])
def test_rejects_paths_ambiguous_for_shared_trainer_and_yaml(tmp_path, name):
    mod = load_script()
    with pytest.raises(ValueError, match="YAML path"):
        mod.prepare_dataset(tmp_path / "unused.zip", tmp_path / name)


def test_bad_release_or_missing_member_leaves_no_dataset(tmp_path, monkeypatch):
    mod = load_script()
    archive = make_release(tmp_path, omit_label=True)
    target = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA-256"):
        mod.prepare_dataset(archive, target)
    assert not target.exists()
    monkeypatch.setattr(mod, "FORMAL_V1_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises((ValueError, KeyError), match="G01P1__after"):
        mod.prepare_dataset(archive, target)
    assert not target.exists()
