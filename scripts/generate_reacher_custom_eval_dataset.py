"""Generate a Reacher custom XML HDF5 eval dataset for a given task."""
import os
import sys
os.environ["MUJOCO_GL_BACKEND"] = "egl"

from mujoco.egl import GLContext as _EGLContext
_egl_ctx = _EGLContext(224, 224)
_egl_ctx.make_current()

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.reacher_custom_env
from stable_worldmodel.data.formats.hdf5 import HDF5Writer

CUSTOM_ENV_ID = "swm/ReacherCustom-v0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--task",
        type=str,
        default="qpos_match",
        choices=["qpos_match", "reach", "push"],
        help="Task mode for the eval env.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. Default: <cache_dir>/datasets/reacher_<task>_eval.h5",
    )
    args = parser.parse_args()

    env = gym.make(
        CUSTOM_ENV_ID,
        task=args.task,
        render_mode="rgb_array",
        width=args.image_size,
        height=args.image_size,
        max_episode_steps=args.max_steps,
    )
    cache_dir = Path(swm.data.utils.get_cache_dir())
    if args.output:
        output = Path(args.output)
    else:
        output = cache_dir / "datasets" / f"reacher_{args.task}_eval.h5"
    output.parent.mkdir(parents=True, exist_ok=True)

    with HDF5Writer(str(output), mode="overwrite") as writer:
        for ep_idx in range(args.episodes):
            frames, actions, qpos_list, qvel_list = [], [], [], []
            env.reset()
            frames.append(env.render())
            qpos_list.append(env.unwrapped.data.qpos.copy())
            qvel_list.append(env.unwrapped.data.qvel.copy())
            for _ in range(args.max_steps):
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                frames.append(env.render())
                actions.append(action)
                qpos_list.append(env.unwrapped.data.qpos.copy())
                qvel_list.append(env.unwrapped.data.qvel.copy())
                if terminated or truncated:
                    break

            n = len(actions)
            episode = {
                "pixels": np.stack(frames[:n]).astype(np.uint8),
                "action": np.stack(actions).astype(np.float32),
                "qpos": np.stack(qpos_list[:n]).astype(np.float32),
                "qvel": np.stack(qvel_list[:n]).astype(np.float32),
                "episode_idx": np.full(n, ep_idx, dtype=np.int32),
                "step_idx": np.arange(n, dtype=np.int32),
            }
            writer.write_episode(episode)
    print(f"Done: {output}")


if __name__ == "__main__":
    main()
