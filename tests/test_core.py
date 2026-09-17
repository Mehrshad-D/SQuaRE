from pathlib import Path

import torch
from torch import nn
import pytest

from pretrained_isolation.config import layerwise_configurations, load_config, selected
from pretrained_isolation.engine import configure_layers, configure_one_layer
from pretrained_isolation.masking import nm_mask, unstructured_mask
from pretrained_isolation.modules import IsolatedLinear
from pretrained_isolation.quantization import fake_quant_symmetric


ROOT = Path(__file__).resolve().parents[1]


def test_fp32_bypass_is_exact():
    x = torch.randn(4, 7)
    assert torch.equal(fake_quant_symmetric(x, 32), x)


def test_two_of_four_mask_is_exact_per_group():
    score = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    mask = nm_mask(score, 2, 4).reshape(3, 2, 4)
    assert torch.all(mask.sum(-1) == 2)


def test_layerwise_nm_masks_have_exact_requested_counts():
    score = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    for n, m in ((2, 4), (4, 8), (3, 8)):
        mask = nm_mask(score, n, m).reshape(4, -1, m)
        assert torch.all(mask.sum(-1) == n)


def test_unstructured_mask_reaches_requested_count():
    score = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    mask = unstructured_mask(score, 0.5)
    assert mask.sum().item() == 50


def test_deit_regex_selects_expected_layers_only():
    cfg = load_config(ROOT / "configs/deit_tiny_imagenet.yaml")
    assert selected("blocks.0.attn.qkv", "linear", cfg)
    assert selected("blocks.11.mlp.fc2", "linear", cfg)
    assert not selected("head", "linear", cfg)
    assert not selected("patch_embed.proj", "conv2d", cfg)


def test_public_benchmark_configs_select_expected_layers():
    deit = load_config(ROOT / "configs/deit_tiny_imagenetv2.yaml")
    swin = load_config(ROOT / "configs/swin_tiny_imagenetv2.yaml")
    resnet = load_config(ROOT / "configs/resnet18_imagenetv2.yaml")
    assert selected("blocks.11.attn.qkv", "linear", deit)
    assert selected("blocks.4.mlp.fc2", "linear", deit)
    assert selected("layers.3.blocks.1.attn.proj", "linear", swin)
    assert selected("layers.0.blocks.0.mlp.fc1", "linear", swin)
    assert selected("layer4.1.conv2", "conv2d", resnet)
    assert not selected("conv1", "conv2d", resnet)
    for cfg in (deit, swin, resnet):
        grid = layerwise_configurations(cfg)
        assert len(grid) == 16
        assert {item["weight_bits"] for item in grid} == {32, 8, 6, 4}
        assert {item["sparsity"] for item in grid} == {"dense", "2to4", "4to8", "3to8"}


def test_public_model_regexes_match_expected_module_counts():
    timm = pytest.importorskip("timm")
    cases = [
        ("deit_tiny_imagenetv2.yaml", 48),
        ("swin_tiny_imagenetv2.yaml", 48),
        ("resnet18_imagenetv2.yaml", 19),
    ]
    for filename, expected in cases:
        cfg = load_config(ROOT / "configs" / filename)
        model = timm.create_model(cfg["model"]["name"], pretrained=False)
        matched = []
        for name, module in model.named_modules():
            kind = "linear" if isinstance(module, nn.Linear) else "conv2d" if isinstance(module, nn.Conv2d) else ""
            if kind and selected(name, kind, cfg):
                matched.append(name)
        assert len(matched) == expected, (filename, matched)


def test_quantization_only_wrapper_has_zero_sparsity():
    source = nn.Linear(8, 4)
    wrapped = IsolatedLinear(source, weight_bits=4, activation_bits=4)
    wrapped.stats_mode = "range"
    wrapped(torch.randn(3, 8))
    wrapped.stats_mode = "off"
    wrapped(torch.randn(3, 8))
    report = wrapped.report()
    assert report["sparsity"] == 0.0
    assert report["nonzero_weights"] == report["num_weights"]


def test_sparse_fp32_wrapper_only_changes_masked_weights():
    source = nn.Linear(8, 4, bias=False)
    wrapped = IsolatedLinear(source, weight_bits=32, activation_bits=32)
    wrapped.mask[:, ::2] = 0
    x = torch.randn(3, 8)
    expected = torch.nn.functional.linear(x, source.weight * wrapped.mask)
    assert torch.allclose(wrapped(x), expected)
    assert wrapped.report()["sparsity"] == 0.5


def test_configure_one_layer_resets_every_other_layer():
    modules = {
        "first": IsolatedLinear(nn.Linear(16, 8), 32, 32),
        "second": IsolatedLinear(nn.Linear(16, 8), 4, 4),
    }
    modules["second"].mask[:, ::2] = 0
    specification = {
        "weight_bits": 6,
        "activation_bits": 6,
        "structure": "nm",
        "n": 3,
        "m": 8,
        "policy": "magnitude",
    }
    configure_one_layer(modules, "first", specification, seed=42)
    assert modules["first"].weight_bits == modules["first"].activation_bits == 6
    assert modules["first"].report()["sparsity"] == 0.625
    assert modules["second"].weight_bits == modules["second"].activation_bits == 32
    assert modules["second"].report()["sparsity"] == 0.0


def test_configure_layers_applies_every_independent_choice():
    modules = {
        "first": IsolatedLinear(nn.Linear(16, 8), 32, 32),
        "second": IsolatedLinear(nn.Linear(16, 8), 32, 32),
    }
    specifications = {
        "first": {
            "weight_bits": 8, "activation_bits": 8,
            "structure": "dense", "policy": "magnitude",
        },
        "second": {
            "weight_bits": 4, "activation_bits": 4,
            "structure": "nm", "n": 3, "m": 8, "policy": "magnitude",
        },
    }
    configure_layers(modules, specifications, seed=42)
    assert modules["first"].weight_bits == 8
    assert modules["first"].report()["sparsity"] == 0.0
    assert modules["second"].weight_bits == 4
    assert modules["second"].report()["sparsity"] == 0.625
