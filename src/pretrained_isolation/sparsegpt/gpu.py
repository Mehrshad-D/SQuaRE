"""Check the selected GPU before launching; no job killing or busy waiting."""
import csv
import io
import os
import subprocess
from ..obc.cli import now


def available_gpu():
    """A point-in-time safety check, not a GPU reservation or a wait loop."""
    def query(fields, kind):
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(raw)) if row]
    devices = query("index,uuid,memory.total,memory.free", "gpu")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    if len(devices) > 1 and not visible.startswith("GPU-") and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Multiple GPUs with ambiguous CUDA ordering; identify the original device by UUID before retrying")
    matches = [r for r in devices if r[0] == visible or (visible.startswith("GPU-") and r[1].startswith(visible))]
    if len(matches) != 1:
        raise RuntimeError("Cannot identify CUDA device 0; check CUDA_VISIBLE_DEVICES before retrying")
    index, uuid, total, free = matches[0]
    processes = query("gpu_uuid,pid", "compute-apps")
    busy = [pid for gpu, pid in processes if gpu == uuid and pid != str(os.getpid())]
    if busy:
        raise RuntimeError(f"GPU {index} is still occupied by compute PIDs {', '.join(busy)}. Wait for those jobs to finish; no retry was launched.")
    if float(free) < 4096:
        raise RuntimeError(f"GPU {index} has only {free} MiB free; require at least 4096 MiB before retrying")
    return {"gpu_index": index, "gpu_uuid": uuid, "total_mib": float(total), "free_mib": float(free),
            "other_compute_pids": busy, "checked_at": now(), "is_reservation": False}
