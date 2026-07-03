"""Generate a LeWM-compatible Lance dataset from MuJoCo Gymnasium envs.

Examples
--------
# HalfCheetah-v5 random policy, 50 episodes
python scripts/generate_mujoco_dataset.py --env HalfCheetah-v4 --episodes 50 --output ~/.stable-wm/halfcheetah_random.lance

# Hopper with a heuristic/random mixed policy
python scripts/generate_mujoco_dataset.py --env Hopper-v4 --episodes 100 --output ~/.stable-wm/hopper_random.lance
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
import torch
from PIL import Image


def render_env(env, render_mode="rgb_array", camera_id=None):
    """Render a frame; adapt to Gymnasium render API."""
    try:
        if camera_id is not None:
            frame = env.render()
        else:
            frame = env.render()
    except Exception:
        frame = env.render()
    if isinstance(frame, list):
        frame = frame[0]
    return np.array(frame)


def resize(frame, size=224):
    """Resize frame to square (size, size) for ViT compatibility."""
    img = Image.fromarray(frame)
    img = img.convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


def make_env(env_name, seed, image_size=224, camera_name=None):
    """Create a renderable Gymnasium MuJoCo environment."""
    env = gym.make(env_name, render_mode="rgb_array")
    if hasattr(env, "unwrapped"):
        env.unwrapped.model.vis.global_.offwidth = image_size
        env.unwrapped.model.vis.global_.offheight = image_size
    env.reset(seed=seed)
    return env


def collect_episode(env, policy, max_steps, image_size, seed):
    """Collect one episode with (pixels, action, observation)."""
    obs, info = env.reset(seed=seed)
    frames, actions, observations, rewards, dones = [], [], [], [], []
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < max_steps:
        action = policy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        frame = resize(render_env(env), size=image_size)
        frames.append(frame)
        actions.append(np.asarray(action, dtype=np.float32))
        observations.append(np.asarray(obs, dtype=np.float32))
        rewards.append(np.float32(reward))
        dones.append(bool(terminated or truncated))
        obs = next_obs
        steps += 1

    # Pass frames as a list so LanceWriter's image-column probe can index
    # the first sample with vals[0] without tripping numpy truthiness.
    episode = {
        "pixels": [f.astype(np.uint8) for f in frames],
        "action": np.stack(actions, axis=0).astype(np.float32),
        "observation": np.stack(observations, axis=0).astype(np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "done": np.array(dones, dtype=bool),
        "episode_idx": np.full(len(frames), -1, dtype=np.int64),
        "step_idx": np.arange(len(frames), dtype=np.int64),
    }
    return episode


def random_policy(env):
    """Return a random action sampler."""
    def policy(obs):
        return env.action_space.sample()
    return policy


def mixed_random_policy(env, frac_zero=0.2):
    """A policy that occasionally outputs zero action (helps learn resets)."""
    rng = np.random.default_rng()
    def policy(obs):
        if rng.random() < frac_zero:
            return np.zeros(env.action_space.shape, dtype=np.float32)
        return env.action_space.sample()
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Gymnasium MuJoCo env id, e.g. HalfCheetah-v4")
    parser.add_argument("--output", required=True, help="Path to output .lance dataset")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--policy", default="random", choices=["random", "mixed"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env, args.seed, image_size=args.image_size)
    if args.policy == "random":
        policy = random_policy(env)
    else:
        policy = mixed_random_policy(env)

    print(f"Collecting {args.episodes} episodes from {args.env} -> {output}")
    with swm.data.get_format("lance").open_writer(output, mode="overwrite") as writer:
        for ep in range(args.episodes):
            episode = collect_episode(
                env, policy, args.max_steps, args.image_size, seed=args.seed + ep
            )
            writer.write_episode(episode)
            print(f"  episode {ep}: {len(episode['action'])} steps")
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
