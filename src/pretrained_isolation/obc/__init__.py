"""Matched full-Hessian and explicit blockwise OBS adapters; see docs/OBC_COMPARISON.md."""

ALGORITHM_VERSION = "matched-obs-2"
UPSTREAM_REVISION = "9b7979bfc9ee20d87db553823a32ee9890beaa99"


def method_label(block_size):
    return (f"OBC-inspired blockwise OBS (B={block_size}, adapted)" if block_size else
            "OBC (matched ExactOBS + bounded DP, adapted)")
