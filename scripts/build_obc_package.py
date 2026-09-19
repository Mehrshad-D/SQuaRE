"""Build the portable v0.8.0 source package without unrelated workspace files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    prefix = "pretrained-isolation-framework-v0.8.0"
    files = [root / name for name in ("README.md", "pyproject.toml", ".gitignore", "docs/OBC_COMPARISON.md")]
    for directory, pattern in (("src", "*.py"), ("configs", "*.yaml"), ("tests", "*.py")):
        files.extend((root / directory).rglob(pattern))
    scripts = ["build_obc_package.py", "profile_obc_imagenetv2.sh", "run_obc_imagenetv2.sh",
               "compare_obc_imagenetv2.sh", "run_full_imagenetv2.sh", "run_layerwise_imagenetv2.sh",
               "run_joint_thresholds_imagenetv2.sh", "run_global_refinement_imagenetv2.sh"]
    files.extend(root / "scripts" / name for name in scripts)
    files.extend((root / "outputs-v2/imagenetv2").glob("*/*_layerwise.json"))
    files.extend((root / "outputs-v4/imagenetv2").glob("*/*_global_refinement.json"))
    files = sorted(set(files))
    content = {str(p.relative_to(root)): p.read_bytes() for p in files}
    manifest = {"version": "0.8.0", "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in content.items()}}
    content["PACKAGE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    destination = root / "dist" / f"{prefix}.zip"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(content.items()):
            info = zipfile.ZipInfo(f"{prefix}/{name}", date_time=(2026, 9, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".zip.sha256").write_text(f"{checksum}  {destination.name}\n")
    print(json.dumps({"package": str(destination), "bytes": destination.stat().st_size,
                      "sha256": checksum, "files": len(content)}, indent=2))


if __name__ == "__main__":
    main()
