from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from PIL import Image
import timm
from timm.data.imagenet_info import ImageNetInfo
from tqdm import tqdm


def build_transform(model):
    data_cfg = timm.data.resolve_model_data_config(model)
    return timm.data.create_transform(**data_cfg, is_training=False), data_cfg


class TinyImageNetValidation(Dataset):
    def __init__(self, val_dir: Path, class_to_idx: dict[str, int], transform):
        annotations = val_dir / "val_annotations.txt"
        images_dir = val_dir / "images"
        if not annotations.is_file() or not images_dir.is_dir():
            raise FileNotFoundError(
                f"Expected Tiny ImageNet files {annotations} and {images_dir}"
            )
        self.samples: list[tuple[Path, int]] = []
        for line in annotations.read_text().splitlines():
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            filename, wnid = fields[:2]
            if wnid not in class_to_idx:
                raise ValueError(f"Validation class {wnid} is absent from the training folders")
            self.samples.append((images_dir / filename, class_to_idx[wnid]))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), target


class CanonicalNumericImageFolder(ImageFolder):
    """ImageFolder whose numeric directory name is the actual class index.

    ImageNetV2 uses directories named 0 through 999.  Torchvision's default
    lexicographic ordering would place class 10 before class 2 and corrupt the
    labels, so this override preserves the canonical ImageNet class index.
    """

    def find_classes(self, directory: str):
        names = [entry.name for entry in os.scandir(directory) if entry.is_dir()]
        if not names or not all(name.isdigit() for name in names):
            raise ValueError(
                f"ImageNetV2 must contain numeric class directories, found: {names[:5]}"
            )
        values = sorted(int(name) for name in names)
        expected = list(range(1000))
        if values != expected:
            raise ValueError(
                f"ImageNetV2 must contain class directories 0..999; found {len(values)} classes"
            )
        classes = [str(value) for value in values]
        return classes, {name: int(name) for name in classes}


def _sha256_manifest(dataset: ImageFolder) -> dict[str, str]:
    root = Path(dataset.root).resolve()
    cache_path = root / ".ptq_sha256_manifest.json"
    cached: dict = {}
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            cached = {}
    old_files = cached.get("files", {}) if cached.get("version") == 1 else {}
    new_files: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for filename, _ in tqdm(
        dataset.samples,
        desc=f"hash-{root.name}",
        unit="image",
        leave=False,
    ):
        path = Path(filename)
        rel = str(path.resolve().relative_to(root))
        stat = path.stat()
        old = old_files.get(rel, {})
        if old.get("size") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
            digest = old.get("sha256")
        else:
            digest = None
        if not digest:
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        new_files[rel] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }
        hashes[str(path)] = digest
    payload = {"version": 1, "root": str(root), "files": new_files}
    try:
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(cache_path)
    except OSError:
        pass
    return hashes


def _exclude_exact_overlap(calibration: ImageFolder, evaluation: ImageFolder) -> int:
    evaluation_hashes = set(_sha256_manifest(evaluation).values())
    calibration_hashes = _sha256_manifest(calibration)
    retained = [
        sample for sample in calibration.samples
        if calibration_hashes[sample[0]] not in evaluation_hashes
    ]
    removed = len(calibration.samples) - len(retained)
    calibration.samples = retained
    calibration.imgs = retained
    calibration.targets = [target for _, target in retained]
    return removed


def _tiny_imagenet_datasets(calib_dir: Path, eval_dir: Path, transform):
    calibration = ImageFolder(calib_dir, transform=transform)
    classes = calibration.classes
    if len(classes) != 200:
        raise ValueError(f"Tiny ImageNet training directory must contain 200 classes, found {len(classes)}")
    if (eval_dir / "images").is_dir() and (eval_dir / "val_annotations.txt").is_file():
        evaluation = TinyImageNetValidation(eval_dir, calibration.class_to_idx, transform)
        validation_layout = "original-images-plus-annotations"
    else:
        evaluation = ImageFolder(eval_dir, transform=transform)
        if evaluation.classes != classes:
            raise ValueError(
                "Reorganized Tiny ImageNet validation class folders do not match training classes"
            )
        validation_layout = "reorganized-class-folders"
    imagenet1k_synsets = ImageNetInfo("imagenet-1k").label_names()
    imagenet1k_index = {wnid: index for index, wnid in enumerate(imagenet1k_synsets)}
    missing = [wnid for wnid in classes if wnid not in imagenet1k_index]
    if missing:
        raise ValueError(f"Tiny ImageNet classes absent from ImageNet-1k output space: {missing[:5]}")
    # ImageFolder targets are positions in sorted(class WNID). Slice the model's
    # 1000 logits in that same order to obtain a correct restricted 200-way task.
    output_indices = [imagenet1k_index[wnid] for wnid in classes]
    label_space = {
        "dataset": "tiny-imagenet-200",
        "evaluation_mode": "restricted-200-from-imagenet1k-logits",
        "num_classes": 200,
        "validation_layout": validation_layout,
        "model_output_indices": output_indices,
        "class_wnids": classes,
    }
    return calibration, evaluation, output_indices, label_space


def make_loaders(model, cfg: dict):
    transform, data_cfg = build_transform(model)
    eval_dir = Path(cfg["eval_dir"]).expanduser()
    calib_dir = Path(cfg["calib_dir"]).expanduser()
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")
    dataset_name = cfg.get("dataset", "imagenet1k").lower()
    if dataset_name in {"imagenet", "imagenet1k", "imagenet-1k"}:
        evaluation = ImageFolder(eval_dir, transform=transform)
        calibration_full = ImageFolder(calib_dir, transform=transform)
        if evaluation.classes != calibration_full.classes:
            raise ValueError("Calibration and evaluation directories have different class mappings")
        output_indices = None
        label_space = {
            "dataset": "ImageNet-1k",
            "evaluation_mode": "full-1000-way",
            "num_classes": len(evaluation.classes),
        }
    elif dataset_name in {"tiny_imagenet200", "tiny-imagenet-200", "tinyimagenet200"}:
        calibration_full, evaluation, output_indices, label_space = _tiny_imagenet_datasets(
            calib_dir, eval_dir, transform
        )
    elif dataset_name in {"imagenetv2", "imagenet-v2"}:
        evaluation = CanonicalNumericImageFolder(eval_dir, transform=transform)
        calibration_full = CanonicalNumericImageFolder(calib_dir, transform=transform)
        overlap_removed = 0
        if cfg.get("exclude_calibration_eval_duplicates", True):
            overlap_removed = _exclude_exact_overlap(calibration_full, evaluation)
        output_indices = None
        label_space = {
            "dataset": "ImageNetV2",
            "evaluation_variant": cfg.get("evaluation_variant", "matched-frequency"),
            "calibration_variant": cfg.get("calibration_variant", "threshold-0.7"),
            "evaluation_mode": "full-1000-way-canonical-numeric-labels",
            "num_classes": 1000,
            "exact_calibration_eval_duplicates_removed": overlap_removed,
            "calibration_pool_after_duplicate_removal": len(calibration_full),
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    requested = int(cfg.get("calibration_samples", 1024))
    count = min(requested, len(calibration_full))
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    indices = torch.randperm(len(calibration_full), generator=generator)[:count].tolist()
    calibration = Subset(calibration_full, indices)
    common = {
        "batch_size": int(cfg.get("batch_size", 64)),
        "num_workers": int(cfg.get("num_workers", 4)),
        "pin_memory": True,
    }
    return (
        DataLoader(calibration, shuffle=False, **common),
        DataLoader(evaluation, shuffle=False, **common),
        data_cfg,
        len(calibration),
        len(evaluation),
        output_indices,
        label_space,
    )
