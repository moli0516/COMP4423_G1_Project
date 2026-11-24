import hashlib
import importlib.util
import json
import subprocess
import sys
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


def make_photo_release(tmp_path, monkeypatch, mod):
    from io import BytesIO

    from PIL import Image

    archive = make_release(tmp_path)
    with zipfile.ZipFile(archive) as src:
        content = {name: src.read(name) for name in src.namelist()}
    buffer = BytesIO()
    Image.new("RGB", (30, 20), "blue").save(buffer, "JPEG")
    fixed = buffer.getvalue()
    for name in content:
        if name.endswith(".jpg"):
            content[name] = fixed + (b"extra" if "/train/" in name else b"")
    with zipfile.ZipFile(archive, "w") as dst:
        for name, data in content.items():
            dst.writestr(name, data)
    monkeypatch.setattr(mod, "FORMAL_V1_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    raw = tmp_path / "raw"
    mod.prepare_dataset(archive, raw)
    receipt = json.loads((raw / "source_receipt.json").read_bytes())
    member_hash = hashlib.sha256(
        json.dumps(receipt["member_sha256"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(mod, "FORMAL_V1_MEMBERS_SHA256", member_hash, raising=False)
    return raw, receipt


def test_training_copy_is_separate_stable_and_verifiable(tmp_path, monkeypatch):
    from PIL import Image

    mod = load_script()
    raw, source = make_photo_release(tmp_path, monkeypatch, mod)

    def check(path):
        p = Path(path)
        if p.read_bytes().endswith(b"extra"):
            p.write_bytes(p.read_bytes()[:-5])
        with Image.open(p) as im:
            return "", (im.height, im.width)

    monkeypatch.setattr(
        mod, "_vision_backend", lambda: (check, {"ultralytics": "fixture"}), raising=False
    )
    one, two = tmp_path / "one", tmp_path / "two"
    assert hasattr(mod, "prepare_training_copy"), (
        "Raw staging lacks an isolated backend-verified training copy"
    )
    mod.prepare_training_copy(raw, one)
    mod.prepare_training_copy(raw, two)
    first = mod.verify_training_copy(one)
    second = mod.verify_training_copy(two)
    assert first["effective_member_sha256"] == second["effective_member_sha256"]
    assert first["source_member_sha256"] == source["member_sha256"]
    assert len(first["effective_member_sha256"]) == 8
    assert first["changed_images"] == [
        "images/train/G06P1__after.jpg",
        "images/train/G06P1__before.jpg",
    ]
    assert set(map(tuple, first["image_wh"].values())) == {(30, 20)}
    assert first["effective_members_sha256"] == second["effective_members_sha256"]
    assert first["local_yaml_sha256"] != second["local_yaml_sha256"]
    assert all(
        hashlib.sha256((raw / name).read_bytes()).hexdigest() == digest
        for name, digest in source["member_sha256"].items()
    )
    # A manifest pointing at a different split must fail even with a matching receipt entry.
    manifest = two / "dataset.yaml"
    manifest.write_bytes(manifest.read_bytes().replace(b"val: images/val", b"val: images/train"))
    second["local_yaml_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (two / "effective_receipt.json").write_text(json.dumps(second))
    with pytest.raises(ValueError, match="dataset.yaml"):
        mod.verify_training_copy(two)
    stale = one / "images/train/G06P1__before.npy"
    stale.write_bytes(b"cached alternate pixels")
    with pytest.raises(ValueError, match="inventory"):
        mod.verify_training_copy(one)
    stale.unlink()
    (one / "labels/val/G01P1__before.txt").write_text("")
    with pytest.raises(ValueError, match="G01P1__before.txt"):
        mod.verify_training_copy(one)


def test_legacy_source_mutation_fails_without_publishing_work(tmp_path, monkeypatch):
    mod = load_script()
    raw, _ = make_photo_release(tmp_path, monkeypatch, mod)
    monkeypatch.setattr(mod, "_vision_backend", lambda: (None, {}), raising=False)
    (raw / "images/train/G06P1__after.jpg").write_bytes(b"previous training altered the source")
    assert hasattr(mod, "prepare_training_copy"), "Existing raw-stage drift must be detected"
    with pytest.raises(ValueError, match="G06P1__after.jpg"):
        mod.prepare_training_copy(raw, tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_pinned_backend_repairs_only_working_jpegs(tmp_path, monkeypatch):
    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip("Install the pinned vision extra for the integration test")
    mod = load_script()
    raw, _ = make_photo_release(tmp_path, monkeypatch, mod)
    work = tmp_path / "work"
    # The backend installs process-wide PIL and OpenCV patches. Exercise it in its own process.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from scripts import prepare_e2_dataset as p

p.FORMAL_V1_SHA256, p.FORMAL_V1_MEMBERS_SHA256 = sys.argv[1:3]
p.prepare_training_copy(Path(sys.argv[3]), Path(sys.argv[4]))
""",
            mod.FORMAL_V1_SHA256,
            mod.FORMAL_V1_MEMBERS_SHA256,
            str(raw),
            str(work),
        ],
        cwd=SCRIPT.parents[1],
        check=True,
    )
    receipt = mod.verify_training_copy(work)
    assert len(receipt["changed_images"]) == 2
    assert (work / receipt["changed_images"][0]).read_bytes().endswith(b"\xff\xd9")
    assert (raw / receipt["changed_images"][0]).read_bytes().endswith(b"extra")
