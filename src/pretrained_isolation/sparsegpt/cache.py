"""Only atomically committed, checksum-verified candidates are reusable."""
import json
from ..obc.evaluation import file_sha256
from ..obc.runtime import atomic_torch, load_tensors
from ..runner import _atomic_json


def candidate_keys(directory):
    path = directory / "candidate_checksums.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_candidate(directory, key, entry, fingerprint):
    entry = {**entry, "run_fingerprint": fingerprint}
    path = directory / f"{key}.pt"
    atomic_torch(path, entry)
    checksums = candidate_keys(directory)
    checksums[key] = file_sha256(path)
    _atomic_json(directory / "candidate_checksums.json", checksums)


def load_candidate(directory, key, fingerprint):
    checksums = candidate_keys(directory)
    path = directory / f"{key}.pt"
    if key not in checksums or file_sha256(path) != checksums[key]:
        raise ValueError(f"Candidate checksum missing or changed: {path}")
    entry = load_tensors(path)
    if entry.get("run_fingerprint") != fingerprint or entry["configuration"] != key:
        raise ValueError("Candidate identity mismatch")
    return entry
