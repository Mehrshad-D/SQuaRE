"""Execute only the pinned upstream class and quantize function for oracle tests.

No transformers installation or global backend-flag changes are needed.
The numerical class body is unmodified; dependency bindings are supplied here.
"""
import ast
import math
from pathlib import Path
import time
from types import SimpleNamespace
import torch
from .core import synchronize


def reference_class():
    vendor = Path(__file__).parent / "vendor"
    nodes = []
    for filename, kind, name in (("quant_reference.py", ast.FunctionDef, "quantize"),
                                 ("sparsegpt_reference.py", ast.ClassDef, "SparseGPT")):
        tree = ast.parse((vendor / filename).read_text())
        nodes.extend(n for n in tree.body if isinstance(n, kind) and n.name == name)
    if len(nodes) != 2:
        raise ValueError("Pinned upstream reference definitions missing")
    # Upstream's unconditional CUDA synchronization is a no-op for CPU tests.
    proxy = SimpleNamespace(**{n: getattr(torch, n) for n in dir(torch)})
    proxy.cuda = SimpleNamespace(synchronize=lambda: synchronize(torch.device("cuda")) if torch.cuda.is_available() else None)
    namespace = {"torch": proxy, "nn": torch.nn, "time": time, "math": math,
                 "DEBUG": False, "transformers": SimpleNamespace(Conv1D=type("UnusedConv1D", (), {}))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned_sparsegpt_reference", "exec"), namespace)
    return namespace["SparseGPT"]


class MatchedQuantizer:
    """Map the shared signed minmax grid into the upstream affine API."""
    def __init__(self, weight, bits):
        qmax = 2 ** (bits - 1) - 1
        self.scale = (weight.flatten(1).float().abs().amax(1, keepdim=True) / qmax).clamp_min(torch.finfo(torch.float32).eps)
        self.zero = torch.full_like(self.scale, qmax)
        self.maxq = 2 * qmax

    def ready(self):
        return True
