"""Generate a Reacher DMControl HDF5 eval dataset (dmc/reacher_random)."""
import os
os.environ["MUJOCO_GL_BACKEND"] = "egl"

from mujoco.egl import GLContext
_egl_ctx = GLContext(224, 224)
_egl_ctx.make_current()

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
from stable_worldmodel.data.formats.hdf5 import HDF5Writer

REACHER_DMC_ID = "swm/ReacherDMControl-v0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the HDF5 file. Default: <cache_dir>/datasets/dmc/reacher_random.h5",
    )
    args = parser.parse_args()

    env = gym.make(
        REACHER_DMC_ID,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )
    cache_dir = Path(swm.data.utils.get_cache_dir())
    output = Path(args.output) if args.output else cache_dir / "datasets" / "dmc" / "reacher_random.h5"
    output.parent.mkdir(parents=True, exist_ok=True)

    with HDF5Writer(str(output), mode="overwrite") as writer:
        for ep_idx in range(args.episodes):
            frames, actions, qpos_list, qvel_list = [], [], [], []
            env.reset()
            physics = env.unwrapped.env.physics
            frames.append(env.render())
            qpos_list.append(physics.data.qpos.copy())
            qvel_list.append(physics.data.qvel.copy())
            for step in range(args.max_steps):
                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                frames.append(env.render())
                actions.append(action)
                physics = env.unwrapped.env.physics
                qpos_list.append(physics.data.qpos.copy())
                qvel_list.append(physics.data.qvel.copy())
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
