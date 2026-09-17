from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
import urllib.request

from tqdm import tqdm


VARIANTS = {
    "matched-frequency": {
        "url": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz",
        "archive": "imagenetv2-matched-frequency.tar.gz",
        "directory": "imagenetv2-matched-frequency-format-val",
    },
    "threshold-0.7": {
        "url": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-threshold0.7.tar.gz",
        "archive": "imagenetv2-threshold0.7.tar.gz",
        "directory": "imagenetv2-threshold0.7-format-val",
    },
}


def _download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(request) as response:
        resumed = response.status == 206 and existing > 0
        if not resumed:
            existing = 0
        total_header = response.headers.get("Content-Length")
        total = existing + int(total_header) if total_header else None
        mode = "ab" if resumed else "wb"
        with partial.open(mode) as stream, tqdm(
            total=total, initial=existing, unit="B", unit_scale=True,
            desc=destination.name,
        ) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                progress.update(len(chunk))
    partial.replace(destination)


def _safe_extract(archive: Path, root: Path) -> None:
    root_resolved = root.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (root / member.name).resolve()
            if root_resolved not in target.parents and target != root_resolved:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not accepted: {member.name}")
        bundle.extractall(root)


def _validate(root: Path) -> tuple[int, int]:
    class_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
    images = sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(class_dirs) != 1000 or images != 10000:
        raise ValueError(
            f"Invalid ImageNetV2 extraction at {root}: {len(class_dirs)} classes, {images} images"
        )
    return len(class_dirs), images


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public ImageNetV2 benchmark")
    parser.add_argument("--root", default="data/imagenetv2")
    parser.add_argument(
        "--variants", nargs="+", choices=sorted(VARIANTS),
        default=["matched-frequency", "threshold-0.7"],
    )
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        info = VARIANTS[variant]
        extracted = root / info["directory"]
        if extracted.is_dir():
            classes, images = _validate(extracted)
            print(f"{variant}: already ready ({classes} classes, {images} images): {extracted}")
            continue
        archive = root / info["archive"]
        if not archive.is_file():
            _download(info["url"], archive)
        print(f"Extracting {archive.name} ...")
        _safe_extract(archive, root)
        classes, images = _validate(extracted)
        print(f"{variant}: ready ({classes} classes, {images} images): {extracted}")
        if not args.keep_archives:
            archive.unlink()


if __name__ == "__main__":
    main()
