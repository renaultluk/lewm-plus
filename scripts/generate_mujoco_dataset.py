"""Generate a LeWM-compatible Lance dataset from MuJoCo Gymnasium envs.

Examples
--------
# HalfCheetah-v5 random policy, 50 episodes
python scripts/generate_mujoco_dataset.py --env HalfCheetah-v5 --episodes 50 --output ~/.stable-wm/halfcheetah_random.lance

# Hopper with a heuristic/random mixed policy
python scripts/generate_mujoco_dataset.py --env Hopper-v5 --episodes 100 --output ~/.stable-wm/hopper_random.lance

# Reacher task-agnostic videos (one task per episode)
python scripts/generate_mujoco_dataset.py --env Reacher-v5 --policy reacher_multitask --episodes 200 --output ~/.stable-wm/reacher_multitask.lance
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
import torch
from PIL import Image, ImageDraw


REACHER_TASKS = [
    "reach_red_spot",
    "push_blue_object_to_blue_spot",
    "push_purple_ball_to_edge",
    "fold_in_on_itself",
    "trace_circle",
]


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


def _sample_workspace_point(rng, radius=0.18):
    angle = rng.uniform(0.0, 2.0 * np.pi)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0))
    return np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)


def _sample_edge_point(rng, radius=0.21):
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32)


def _fk_reacher(q, link_length=0.1):
    q1, q2 = float(q[0]), float(q[1])
    x = link_length * np.cos(q1) + link_length * np.cos(q1 + q2)
    y = link_length * np.sin(q1) + link_length * np.sin(q1 + q2)
    return np.array([x, y], dtype=np.float32)


def _ee_controller(q, goal_xy, link_length=0.1, gain=18.0):
    q1, q2 = float(q[0]), float(q[1])
    ee = _fk_reacher(q, link_length=link_length)
    delta = goal_xy - ee

    j11 = -link_length * np.sin(q1) - link_length * np.sin(q1 + q2)
    j12 = -link_length * np.sin(q1 + q2)
    j21 = link_length * np.cos(q1) + link_length * np.cos(q1 + q2)
    j22 = link_length * np.cos(q1 + q2)
    jacobian = np.array([[j11, j12], [j21, j22]], dtype=np.float32)
    action = gain * jacobian.T @ delta
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _set_reacher_target(env, target_xy):
    qpos = env.unwrapped.data.qpos.copy()
    qvel = env.unwrapped.data.qvel.copy()
    if qpos.shape[0] >= 4:
        qpos[-2:] = target_xy
        env.unwrapped.set_state(qpos, qvel)


def _world_to_px(xy, size):
    scale = size / 0.52
    cx = (size * 0.5) + (xy[0] * scale)
    cy = (size * 0.5) - (xy[1] * scale)
    return int(np.clip(cx, 0, size - 1)), int(np.clip(cy, 0, size - 1))


def _draw_disk(draw, center_px, radius_px, color):
    x, y = center_px
    draw.ellipse((x - radius_px, y - radius_px, x + radius_px, y + radius_px), fill=color)


def _annotate_reacher_task(frame, image_size, task_state):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    if "target" in task_state:
        color = task_state.get("target_color", (255, 32, 32))
        _draw_disk(draw, _world_to_px(task_state["target"], image_size), 6, color)

    if "object" in task_state:
        color = task_state.get("object_color", (35, 120, 255))
        _draw_disk(draw, _world_to_px(task_state["object"], image_size), 8, color)

    if "edge_goal" in task_state:
        _draw_disk(draw, _world_to_px(task_state["edge_goal"], image_size), 6, (190, 70, 210))

    return np.array(img)


def _init_reacher_task(env, task_name, rng):
    if task_name == "reach_red_spot":
        target = _sample_workspace_point(rng)
        _set_reacher_target(env, target)
        return {"name": task_name, "target": target, "target_color": (240, 40, 40)}

    if task_name == "push_blue_object_to_blue_spot":
        obj = _sample_workspace_point(rng)
        target = _sample_workspace_point(rng)
        _set_reacher_target(env, np.array([0.25, 0.25], dtype=np.float32))
        return {
            "name": task_name,
            "object": obj,
            "target": target,
            "object_color": (35, 125, 255),
            "target_color": (110, 175, 255),
        }

    if task_name == "push_purple_ball_to_edge":
        obj = _sample_workspace_point(rng, radius=0.12)
        edge_goal = _sample_edge_point(rng)
        _set_reacher_target(env, np.array([0.25, 0.25], dtype=np.float32))
        return {
            "name": task_name,
            "object": obj,
            "edge_goal": edge_goal,
            "object_color": (170, 70, 210),
            "target_color": (220, 145, 235),
        }

    if task_name == "fold_in_on_itself":
        _set_reacher_target(env, np.array([0.25, 0.25], dtype=np.float32))
        return {
            "name": task_name,
            "joint_goal": np.array([1.6, -2.5], dtype=np.float32),
        }

    if task_name == "trace_circle":
        center = _sample_workspace_point(rng, radius=0.06)
        _set_reacher_target(env, center)
        return {
            "name": task_name,
            "center": center,
            "radius": float(rng.uniform(0.04, 0.09)),
            "omega": float(rng.uniform(0.05, 0.11)),
        }

    raise ValueError(f"Unknown reacher task: {task_name}")


def _update_virtual_object(ee_xy, obj_xy, contact_radius=0.05):
    diff = ee_xy - obj_xy
    dist = float(np.linalg.norm(diff))
    if dist < contact_radius:
        obj_xy = obj_xy + 0.33 * diff
    return np.clip(obj_xy, -0.24, 0.24)


def _reacher_task_action(env, task_state, step_idx):
    q = env.unwrapped.data.qpos[:2].copy()
    ee_xy = _fk_reacher(q)
    name = task_state["name"]

    if name == "reach_red_spot":
        return _ee_controller(q, task_state["target"]), ee_xy

    if name == "push_blue_object_to_blue_spot":
        obj = task_state["object"]
        target = task_state["target"]
        push_dir = target - obj
        norm = float(np.linalg.norm(push_dir))
        push_hat = push_dir / (norm + 1e-6)
        approach = obj - 0.035 * push_hat
        contact = float(np.linalg.norm(ee_xy - obj)) < 0.05
        goal = target if contact else approach
        return _ee_controller(q, goal), ee_xy

    if name == "push_purple_ball_to_edge":
        obj = task_state["object"]
        edge_goal = task_state["edge_goal"]
        push_dir = edge_goal - obj
        norm = float(np.linalg.norm(push_dir))
        push_hat = push_dir / (norm + 1e-6)
        approach = obj - 0.035 * push_hat
        contact = float(np.linalg.norm(ee_xy - obj)) < 0.05
        goal = edge_goal if contact else approach
        return _ee_controller(q, goal), ee_xy

    if name == "fold_in_on_itself":
        q_goal = task_state["joint_goal"]
        action = 2.2 * (q_goal - q)
        return np.clip(action, -1.0, 1.0).astype(np.float32), ee_xy

    if name == "trace_circle":
        center = task_state["center"]
        radius = task_state["radius"]
        omega = task_state["omega"]
        phase = step_idx * omega
        goal = center + radius * np.array([np.cos(phase), np.sin(phase)], dtype=np.float32)
        return _ee_controller(q, goal), ee_xy

    raise ValueError(f"Unknown reacher task: {name}")


def collect_reacher_multitask_episode(env, max_steps, image_size, seed, tasks):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    task_name = str(rng.choice(tasks))
    task_id = tasks.index(task_name)
    task_state = _init_reacher_task(env, task_name, rng)

    frames, actions, observations, rewards, dones = [], [], [], [], []
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < max_steps:
        action, ee_xy = _reacher_task_action(env, task_state, step_idx=steps)
        next_obs, reward, terminated, truncated, info = env.step(action)

        if "object" in task_state:
            task_state["object"] = _update_virtual_object(ee_xy, task_state["object"])

        frame = resize(render_env(env), size=image_size)
        frame = _annotate_reacher_task(frame, image_size=image_size, task_state=task_state)
        frames.append(frame)
        actions.append(np.asarray(action, dtype=np.float32))
        observations.append(np.asarray(obs, dtype=np.float32))
        rewards.append(np.float32(reward))
        dones.append(bool(terminated or truncated))
        obs = next_obs
        steps += 1

    episode = {
        "pixels": [f.astype(np.uint8) for f in frames],
        "action": np.stack(actions, axis=0).astype(np.float32),
        "observation": np.stack(observations, axis=0).astype(np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "done": np.array(dones, dtype=bool),
        "task_id": np.full(len(frames), task_id, dtype=np.int64),
    }
    return episode, task_name


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
    parser.add_argument("--env", required=True, help="Gymnasium MuJoCo env id, e.g. HalfCheetah-v5")
    parser.add_argument("--output", required=True, help="Path to output .lance dataset")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--policy",
        default="random",
        choices=["random", "mixed", "reacher_multitask"],
        help="Policy used for data collection.",
    )
    parser.add_argument(
        "--reacher_tasks",
        default=",".join(REACHER_TASKS),
        help="Comma-separated Reacher tasks used by --policy reacher_multitask.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env, args.seed, image_size=args.image_size)
    policy = None
    if args.policy == "random":
        policy = random_policy(env)
    elif args.policy == "mixed":
        policy = mixed_random_policy(env)
    else:
        if "Reacher" not in args.env:
            raise ValueError("--policy reacher_multitask requires a Reacher MuJoCo env (e.g., Reacher-v5).")
        reacher_tasks = [t.strip() for t in args.reacher_tasks.split(",") if t.strip()]
        invalid = [t for t in reacher_tasks if t not in REACHER_TASKS]
        if invalid:
            raise ValueError(f"Unknown reacher task(s): {invalid}. Supported: {REACHER_TASKS}")
        if not reacher_tasks:
            raise ValueError("--reacher_tasks must contain at least one task.")

    print(f"Collecting {args.episodes} episodes from {args.env} -> {output}")
    with swm.data.get_format("lance").open_writer(output, mode="overwrite") as writer:
        for ep in range(args.episodes):
            if args.policy == "reacher_multitask":
                episode, task_name = collect_reacher_multitask_episode(
                    env,
                    max_steps=args.max_steps,
                    image_size=args.image_size,
                    seed=args.seed + ep,
                    tasks=reacher_tasks,
                )
            else:
                episode = collect_episode(
                    env, policy, args.max_steps, args.image_size, seed=args.seed + ep
                )
                task_name = "n/a"
            writer.write_episode(episode)
            print(f"  episode {ep}: {len(episode['action'])} steps task={task_name}")
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
