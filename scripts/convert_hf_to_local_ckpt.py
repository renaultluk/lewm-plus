"""Convert official HF weights.pt into a load_pretrained-compatible folder.

The resulting folder contains `weights.pt` (remapped state dict) and
`config.json` and can be used directly with:

    python eval.py --config-name=pusht.yaml policy=pusht/lewm
"""

import argparse
import json
import re
from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf


def remap_vit_keys(state_dict: dict) -> dict:
    """Remap standard HF ViT keys to the flattened LeWM ViT layout."""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("encoder.embeddings.") or k.startswith("encoder.layernorm."):
            new_sd[k] = v
            continue

        m = re.match(r"encoder\.encoder\.layer\.(\d+)\.(.*)", k)
        if not m:
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
        new_sd[f"encoder.layers.{layer}.{rest}"] = v
    return new_sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    cfg = json.loads((args.src / "config.json").read_text())
    model = hydra.utils.instantiate(cfg)

    sd = torch.load(args.src / "weights.pt", map_location="cpu", weights_only=False)
    sd = remap_vit_keys(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("WARNING missing keys:", missing[:5])
    if unexpected:
        print("WARNING unexpected keys:", unexpected[:5])

    swm.wm.utils.save_pretrained(
        model,
        run_name=args.run_name,
        config=OmegaConf.create(cfg),
        filename="weights.pt",
    )
    print(f"Saved checkpoint to checkpoints/{args.run_name}")


if __name__ == "__main__":
    main()
