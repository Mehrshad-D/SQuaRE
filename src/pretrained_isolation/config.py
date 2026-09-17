from __future__ import annotations

from pathlib import Path
import re
import yaml


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    required = ["model", "data", "selection", "quantization_sweep", "sparsity_sweep"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    return cfg


def layerwise_configurations(cfg: dict) -> list[dict]:
    """Expand the requested precision x sparsity grid in a stable order."""
    sweep = cfg.get("layerwise_sweep")
    if not sweep:
        raise ValueError("Missing config section: layerwise_sweep")
    quantization = sweep.get("quantization", [])
    sparsity = sweep.get("sparsity", [])
    if not quantization or not sparsity:
        raise ValueError("layerwise_sweep requires non-empty quantization and sparsity lists")

    result = []
    ids = set()
    for quant in quantization:
        bits = int(quant["bits"])
        if bits != 32 and bits < 2:
            raise ValueError(f"Invalid layer-wise precision: {bits}")
        for sparse in sparsity:
            structure = sparse["structure"]
            if structure not in {"dense", "nm"}:
                raise ValueError(f"Unknown layer-wise sparsity structure: {structure}")
            item = {
                "id": f"{quant['id']}__{sparse['id']}",
                "quantization": quant["id"],
                "weight_bits": bits,
                "activation_bits": bits,
                "sparsity": sparse["id"],
                "structure": structure,
                "policy": sweep.get("policy", "magnitude"),
            }
            if structure == "nm":
                item["n"] = int(sparse["n"])
                item["m"] = int(sparse["m"])
                if not 0 < item["n"] < item["m"]:
                    raise ValueError("Layer-wise N:M sparsity requires 0 < N < M")
            if item["id"] in ids:
                raise ValueError(f"Duplicate layer-wise configuration id: {item['id']}")
            ids.add(item["id"])
            result.append(item)
    if len(result) != 16:
        raise ValueError(f"Expected exactly 16 layer-wise configurations, found {len(result)}")
    return result


def selected(name: str, module_type: str, cfg: dict) -> bool:
    selection = cfg["selection"]
    if module_type not in selection.get("module_types", ["linear"]):
        return False
    includes = selection.get("include", [".*"])
    excludes = selection.get("exclude", [])
    return any(re.fullmatch(p, name) for p in includes) and not any(
        re.fullmatch(p, name) for p in excludes
    )
