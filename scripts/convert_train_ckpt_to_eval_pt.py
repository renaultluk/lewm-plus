"""Convert a training .ckpt into an eval-ready .pt policy folder.

This script extracts model weights from a Lightning/stable-pretraining checkpoint,
instantiates the LeWM model, and writes a stable-worldmodel-compatible policy
folder under `${STABLEWM_HOME}/checkpoints/<run_name>/`.

Output files:
  - weights.pt
  - config.json

Example:
  python scripts/convert_train_ckpt_to_eval_pt.py \
      --src-ckpt /project/<group>/lewm-plus/.stable-wm/checkpoints/pusht_h5_replicate_run/pusht_h5_replicate_weights.ckpt \
      --run-name pusht_h5_replicate_eval

Then evaluate with:
  policy=pusht_h5_replicate_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repo root is importable when this script is executed as
# `python scripts/convert_train_ckpt_to_eval_pt.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import stable_worldmodel as swm
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


def _load_model_state_dict(src_ckpt: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(src_ckpt, map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)}")

    model_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_sd[k[len("model.") :]] = v
        else:
            model_sd[k] = v
    return model_sd


def _build_model_cfg(args, action_input_dim: int):
    if args.config is not None:
        cfg = OmegaConf.load(args.config)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        train_cfg_dir = repo_root / "config" / "train"
        with initialize_config_dir(config_dir=str(train_cfg_dir), version_base=None):
            cfg = compose(config_name=args.train_config_name, overrides=[f"data={args.data}"])

    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = int(action_input_dim)

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-ckpt", required=True, type=Path, help="Source .ckpt path")
    parser.add_argument("--run-name", required=True, help="Output run_name under checkpoints/")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional saved train config.yaml for exact reconstruction",
    )
    parser.add_argument("--train-config-name", default="lewm")
    parser.add_argument("--data", default="pusht_h5")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional stable-worldmodel cache root override (defaults to STABLEWM_HOME/default)",
    )
    args = parser.parse_args()

    src_ckpt = args.src_ckpt.expanduser().resolve()
    if not src_ckpt.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {src_ckpt}")

    model_sd = _load_model_state_dict(src_ckpt)

    action_key = "action_encoder.patch_embed.weight"
    if action_key not in model_sd:
        raise KeyError(f"Missing key in checkpoint state_dict: {action_key}")
    action_input_dim = model_sd[action_key].shape[1]

    cfg = _build_model_cfg(args, action_input_dim=action_input_dim)
    model = hydra.utils.instantiate(cfg.model)

    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing or unexpected:
        print(f"WARNING: missing keys: {len(missing)}")
        if missing:
            print("  sample:", missing[:10])
        print(f"WARNING: unexpected keys: {len(unexpected)}")
        if unexpected:
            print("  sample:", unexpected[:10])

    swm.wm.utils.save_pretrained(
        model,
        run_name=args.run_name,
        config=cfg.model,
        filename="weights.pt",
        cache_dir=args.cache_dir,
    )

    ckpt_dir = swm.data.utils.get_cache_dir(args.cache_dir, sub_folder="checkpoints") / args.run_name
    print(f"Saved eval policy to: {ckpt_dir}")
    print(f"Use eval override: policy={args.run_name}")


if __name__ == "__main__":
    main()
