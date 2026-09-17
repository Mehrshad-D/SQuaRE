from pretrained_isolation.analyze_layerwise import _best


def _row(configuration, top1, bits, sparsity):
    return {
        "model": "model",
        "layer_index": 0,
        "target_layer": "layer",
        "configuration": configuration,
        "top1": top1,
        "weight_bits": bits,
        "actual_weight_sparsity": sparsity,
        "status": "succeeded",
    }


def test_best_exports_include_all_and_compressed_choices():
    rows = [
        _row("FP32__dense", 70.0, 32, 0.0),
        _row("INT8__dense", 69.8, 8, 0.0),
        _row("INT6__2to4", 69.8, 6, 0.5),
    ]
    assert _best(rows, compressed_only=False)[0]["configuration"] == "FP32__dense"
    assert _best(rows, compressed_only=True)[0]["configuration"] == "INT6__2to4"
