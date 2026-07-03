"""Convert official LeWM HuggingFace weights.pt to eval-ready *_object.ckpt.

The HF checkpoints were saved with a standard HuggingFace ViT key layout
(encoder.encoder.layer...), while the released stable_worldmodel LeWM code
expects a flattened ViT layout (encoder.layers...).  This script remaps the
weights, loads them into a LeWM instance, and pickles the full module so
eval.py / AutoCostModel can use it directly.

Example
-------
python scripts/convert_hf_to_object_ckpt.py \
    --src /path/to/hf_pusht \
    --dst /home/user/.stable-wm/checkpoints/pusht/lewm_object.ckpt
"""

import argparse
import re
from pathlib import Path

import hydra
import torch


def remap_vit_keys(state_dict: dict) -> dict:
    """Remap standard HF ViT keys to the flattened LeWM ViT layout."""
    new_sd = {}
    for k, v in state_dict.items():
        # Embeddings are identical
        if k.startswith("encoder.embeddings.") or k == "encoder.layernorm.weight" or k == "encoder.layernorm.bias":
            new_sd[k] = v
            continue

        # Map encoder.encoder.layer.N.* -> encoder.layers.N.*
        m = re.match(r"encoder\.encoder\.layer\.(\d+)\.(.*)", k)
        if not m:
            # Non-encoder keys (predictor, projectors, etc.) pass through
            new_sd[k] = v
            continue

        layer, rest = m.group(1), m.group(2)
        rest = (
            rest.replace("attention.attention.query", "attention.q_proj")
            .replace("attention.attention.key", "attention.k_proj")
            .replace("attention.attention.value", "attention.v_proj")
            .replace("attention.output.dense", "attention.o_proj")
            .replace("intermediate.dense", "mlp.fc1")
            .replace("output.dense", "mlp.fc2")
        )
        new_k = f"encoder.layers.{layer}.{rest}"
        new_sd[new_k] = v
    return new_sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path, help="HF folder with weights.pt + config.json")
    parser.add_argument("--dst", required=True, type=Path, help="Output *_object.ckpt path")
    args = parser.parse_args()

    cfg = json.loads((args.src / "config.json").read_text())
    model = hydra.utils.instantiate(cfg)

    sd = torch.load(args.src / "weights.pt", map_location="cpu", weights_only=False)
    sd = remap_vit_keys(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("WARNING missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("WARNING unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.dst)
    print(f"Saved {args.dst} ({args.dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    import json
    main()
