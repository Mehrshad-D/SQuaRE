"""Build an additive patch for the original, immutable v0.9.0 source ZIP."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "dist/pretrained-isolation-framework-v0.9.0.zip"
    expected_base = "a2811e7c91acfeb2d58566027c2af245e46b7f0bc238ca8fc8c923309cfbb9ff"
    if hashlib.sha256(base.read_bytes()).hexdigest() != expected_base:
        raise ValueError("Need the original validated v0.9.0 source ZIP")
    with zipfile.ZipFile(base) as z:
        old = json.loads(z.read("pretrained-isolation-framework-v0.9.0/PACKAGE_MANIFEST.json"))
    for name, value in old["files"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != value:
            raise ValueError(f"Original v0.9.0 file changed: {name}")
    files = [p for p in (root / "src/pretrained_isolation/sparsegpt").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    files += [root / name for name in (
        "scripts/launch_sparsegpt.py", "scripts/run_sparsegpt_v4_2h.sh", "scripts/build_sparsegpt_patch.py",
        "SPARSEGPT_COMPARISON.md", "tests/test_sparsegpt.py", "tests/test_sparsegpt_study.py")]
    content = {str(p.relative_to(root)): p.read_bytes() for p in sorted(files)}
    if set(content) & set(old["files"]):
        raise ValueError("Patch must not replace any original package files")
    manifest = {"patch_version": 1, "requires_base_version": "0.9.0", "original_archive_sha256": expected_base,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "files": {name: hashlib.sha256(value).hexdigest() for name, value in content.items()}}
    content["SPARSEGPT_PATCH_MANIFEST.json"] = (json.dumps(manifest, indent=2)+"\n").encode()
    destination = root / "dist/sparsegpt-v0.9.0-patch-v1.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, value in sorted(content.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, value)
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.zip.sha256').write_text(f"{checksum}  {destination.name}\n")
    print(json.dumps({"archive": str(destination), "sha256": checksum, "bytes": destination.stat().st_size,
                      "original_files_unchanged": len(old["files"]), "patch_files": len(content)}, indent=2))


if __name__ == "__main__":
    main()
