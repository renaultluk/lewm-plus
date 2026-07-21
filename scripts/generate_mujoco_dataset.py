"""Generate a LeWM-compatible Lance dataset from MuJoCo Gymnasium envs.

Examples
--------
# HalfCheetah-v5 random policy, 50 episodes
python scripts/generate_mujoco_dataset.py --env HalfCheetah-v5 --episodes 50 --output ~/.stable-wm/halfcheetah_random.lance

# Hopper with a heuristic/random mixed policy
python scripts/generate_mujoco_dataset.py --env Hopper-v5 --episodes 100 --output ~/.stable-wm/hopper_random.lance

# Reacher task-agnostic videos using custom XML scene with objects
python scripts/generate_mujoco_dataset.py --env ReacherTaskAgnostic-v0 --policy reacher_multitask --episodes 200 --output ~/.stable-wm/reacher_multitask.lance
"""

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
from PIL import Image


REACHER_TASKS = [
    "reach_red_spot",
    "push_blue_object_to_blue_spot",
    "push_purple_ball_to_edge",
    "fold_in_on_itself",
    "trace_circle",
]

CUSTOM_REACHER_ENV_ID = "ReacherTaskAgnostic-v0"
CUSTOM_REACHER_XML = Path(__file__).resolve().parent.parent / "assets" / "reacher_task_agnostic.xml"
REACHER_LINK_1 = 0.1
REACHER_LINK_2 = 0.11


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


def _sample_edge_point(rng, radius=0.245):
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32)


def _sample_push_layout(rng):
    for _ in range(64):
        goal = _sample_workspace_point(rng, radius=0.14)
        obj = _sample_workspace_point(rng, radius=0.11)
        d = float(np.linalg.norm(goal - obj))
        if 0.09 <= d <= 0.18:
            return obj, goal
    return obj, goal


def _sample_purple_layout(rng):
    for _ in range(64):
        obj = _sample_workspace_point(rng, radius=0.10)
        edge = _sample_edge_point(rng, radius=0.21)
        d = float(np.linalg.norm(edge - obj))
        if 0.08 <= d <= 0.18:
            return obj, edge
    return obj, edge


def _fk_reacher(q, link1=REACHER_LINK_1, link2=REACHER_LINK_2):
    q1, q2 = float(q[0]), float(q[1])
    x = link1 * np.cos(q1) + link2 * np.cos(q1 + q2)
    y = link1 * np.sin(q1) + link2 * np.sin(q1 + q2)
    return np.array([x, y], dtype=np.float32)


def _ik_reacher(goal_xy, link1=REACHER_LINK_1, link2=REACHER_LINK_2, elbow_sign=1.0):
    x, y = float(goal_xy[0]), float(goal_xy[1])
    r2 = (x * x) + (y * y)
    c2 = (r2 - (link1 * link1) - (link2 * link2)) / (2.0 * link1 * link2)
    c2 = float(np.clip(c2, -1.0, 1.0))
    s2 = elbow_sign * np.sqrt(max(0.0, 1.0 - (c2 * c2)))
    q2 = float(np.arctan2(s2, c2))
    q1 = float(np.arctan2(y, x) - np.arctan2(link2 * s2, link1 + (link2 * c2)))
    return np.array([q1, q2], dtype=np.float32)


def _clamp_to_reacher_workspace(xy, max_radius=0.195):
    r = float(np.linalg.norm(xy))
    if r <= max_radius:
        return xy.astype(np.float32)
    return (xy / (r + 1e-6) * max_radius).astype(np.float32)


def _wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _angle_diff(target, current):
    return _wrap_angle(target - current)


def _ee_controller(q, goal_xy, qvel=None, link1=REACHER_LINK_1, link2=REACHER_LINK_2, gain=18.0):
    q1, q2 = float(q[0]), float(q[1])
    ee = _fk_reacher(q, link1=link1, link2=link2)
    delta = goal_xy - ee

    j11 = -link1 * np.sin(q1) - link2 * np.sin(q1 + q2)
    j12 = -link2 * np.sin(q1 + q2)
    j21 = link1 * np.cos(q1) + link2 * np.cos(q1 + q2)
    j22 = link2 * np.cos(q1 + q2)
    jacobian = np.array([[j11, j12], [j21, j22]], dtype=np.float32)
    action = gain * jacobian.T @ delta
    if qvel is not None:
        action = action - (0.9 * np.asarray(qvel, dtype=np.float32))
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _ee_joint_controller(q, goal_xy, qvel=None, gain=2.6, damping=0.8):
    q_goal_a = _ik_reacher(goal_xy, elbow_sign=1.0)
    q_goal_b = _ik_reacher(goal_xy, elbow_sign=-1.0)
    err_a = _angle_diff(q_goal_a, q)
    err_b = _angle_diff(q_goal_b, q)
    err = err_a if float(np.linalg.norm(err_a)) <= float(np.linalg.norm(err_b)) else err_b
    action = gain * err
    if qvel is not None:
        action = action - (damping * np.asarray(qvel, dtype=np.float32))
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _init_reacher_task(env, task_name, rng):
    target = _sample_workspace_point(rng, radius=0.18)
    blue_obj, blue_goal = _sample_push_layout(rng)
    purple_obj, purple_edge = _sample_purple_layout(rng)

    qpos = env.unwrapped.data.qpos.copy()
    qvel = env.unwrapped.data.qvel.copy()
    _set_body_xy_from_joint(qpos, env, "target", "target_x", "target_y", target)
    _set_body_xy_from_joint(qpos, env, "blue_goal", "blue_goal_x", "blue_goal_y", blue_goal)
    _set_body_xy_from_joint(qpos, env, "blue_object", "blue_obj_x", "blue_obj_y", blue_obj)
    _set_body_xy_from_joint(qpos, env, "purple_ball", "purple_x", "purple_y", purple_obj)
    qvel[:2] = 0.0

    if task_name == "reach_red_spot":
        away = -target
        if float(np.linalg.norm(away)) < 0.06:
            away = target + np.array([0.12, -0.08], dtype=np.float32)
        start_tip = _clamp_to_reacher_workspace(away)
        qpos[:2] = _ik_reacher(start_tip, elbow_sign=1.0 if rng.random() < 0.5 else -1.0)
        env.unwrapped.set_state(qpos, qvel)
        return {"name": task_name, "settle_radius": 0.014}

    if task_name == "push_blue_object_to_blue_spot":
        push_dir = blue_goal - blue_obj
        push_hat = push_dir / (float(np.linalg.norm(push_dir)) + 1e-6)
        start_tip = blue_obj - 0.035 * push_hat
        qpos[:2] = _ik_reacher(start_tip, elbow_sign=1.0 if rng.random() < 0.5 else -1.0)
        env.unwrapped.set_state(qpos, qvel)
        return {"name": task_name, "in_contact": False}

    if task_name == "push_purple_ball_to_edge":
        qpos[:2] = _ik_reacher(purple_obj, elbow_sign=1.0 if rng.random() < 0.5 else -1.0)
        env.unwrapped.set_state(qpos, qvel)
        return {
            "name": task_name,
            "edge_goal": purple_edge,
            "in_contact": False,
        }

    if task_name == "fold_in_on_itself":
        q_goal = np.array([1.6, -2.5], dtype=np.float32)
        if rng.random() < 0.5:
            q_goal = -q_goal
        qpos[:2] = np.array([-0.9, 1.2], dtype=np.float32)
        env.unwrapped.set_state(qpos, qvel)
        return {"name": task_name, "joint_goal": q_goal, "settle_err": 0.05}

    if task_name == "trace_circle":
        qpos[:2] = _ik_reacher(target, elbow_sign=1.0 if rng.random() < 0.5 else -1.0)
        env.unwrapped.set_state(qpos, qvel)
        return {
            "name": task_name,
            "radius": float(rng.uniform(0.03, 0.09)),
            "omega": float(rng.uniform(0.05, 0.14)),
            "phase0": float(rng.uniform(0.0, 2.0 * np.pi)),
        }

    raise ValueError(f"Unknown reacher task: {task_name}")


def _joint_controller(q, q_goal, qvel=None, gain=1.2, damping=0.7):
    action = gain * (q_goal - q)
    if qvel is not None:
        action = action - (damping * qvel)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _joint_qpos_index(env, joint_name):
    return int(env.unwrapped.model.joint(joint_name).qposadr[0])


def _set_body_xy_from_joint(qpos, env, body_name, joint_x, joint_y, xy):
    body_base_xy = np.asarray(env.unwrapped.model.body(body_name).pos[:2], dtype=np.float32)
    qpos[_joint_qpos_index(env, joint_x)] = float(xy[0] - body_base_xy[0])
    qpos[_joint_qpos_index(env, joint_y)] = float(xy[1] - body_base_xy[1])


def _get_body_xy(env, body_name):
    return np.asarray(env.unwrapped.data.body(body_name).xpos[:2], dtype=np.float32)


def _has_geom_contact(env, geom_a, geom_b):
    target = {geom_a, geom_b}
    for i in range(int(env.unwrapped.data.ncon)):
        c = env.unwrapped.data.contact[i]
        g1 = env.unwrapped.model.geom(c.geom1).name
        g2 = env.unwrapped.model.geom(c.geom2).name
        if {g1, g2} == target:
            return True
    return False


def _reacher_task_action(env, task_state, rng, step_idx, prev_action):
    q = np.asarray(env.unwrapped.data.qpos[:2], dtype=np.float32)
    qvel = np.asarray(env.unwrapped.data.qvel[:2], dtype=np.float32)
    ee_xy = _get_body_xy(env, "fingertip")
    target_xy = _get_body_xy(env, "target")
    name = task_state["name"]

    if name == "reach_red_spot":
        if float(np.linalg.norm(target_xy - ee_xy)) < float(task_state.get("settle_radius", 0.014)):
            return np.zeros(2, dtype=np.float32)
        return _ee_joint_controller(q, target_xy, qvel=qvel, gain=2.8, damping=0.9)

    if name == "push_blue_object_to_blue_spot":
        obj_xy = _get_body_xy(env, "blue_object")
        goal_xy = _get_body_xy(env, "blue_goal")
        dist_to_goal = float(np.linalg.norm(goal_xy - obj_xy))
        if dist_to_goal < 0.02:
            return np.zeros(2, dtype=np.float32)
        if _has_geom_contact(env, "fingertip", "blue_object_geom"):
            task_state["in_contact"] = True
        push_dir = goal_xy - obj_xy
        push_hat = push_dir / (float(np.linalg.norm(push_dir)) + 1e-6)
        tip_to_obj = float(np.linalg.norm(ee_xy - obj_xy))
        if (not task_state.get("in_contact", False)) and tip_to_obj < 0.040:
            task_state["in_contact"] = True
        if task_state.get("in_contact", False) and tip_to_obj > 0.06:
            task_state["in_contact"] = False
        if task_state.get("in_contact", False):
            desired_tip = obj_xy + (0.016 * push_hat)
            gain = 2.4
        else:
            desired_tip = obj_xy - (0.044 * push_hat)
            gain = 2.9
        return _ee_joint_controller(q, desired_tip, qvel=qvel, gain=gain, damping=0.9)

    if name == "push_purple_ball_to_edge":
        obj_xy = _get_body_xy(env, "purple_ball")
        edge_goal = task_state["edge_goal"]
        dist_to_goal = float(np.linalg.norm(edge_goal - obj_xy))
        if dist_to_goal < 0.02:
            return np.zeros(2, dtype=np.float32)
        if _has_geom_contact(env, "fingertip", "purple_ball_geom"):
            task_state["in_contact"] = True
        push_dir = edge_goal - obj_xy
        push_hat = push_dir / (float(np.linalg.norm(push_dir)) + 1e-6)
        tip_to_obj = float(np.linalg.norm(ee_xy - obj_xy))
        if (not task_state.get("in_contact", False)) and tip_to_obj < 0.040:
            task_state["in_contact"] = True
        if task_state.get("in_contact", False) and tip_to_obj > 0.06:
            task_state["in_contact"] = False
        if task_state.get("in_contact", False):
            desired_tip = obj_xy + (0.016 * push_hat)
            gain = 2.4
        else:
            desired_tip = obj_xy - (0.044 * push_hat)
            gain = 2.9
        return _ee_joint_controller(q, desired_tip, qvel=qvel, gain=gain, damping=0.9)

    if name == "fold_in_on_itself":
        err = task_state["joint_goal"] - q
        if float(np.linalg.norm(err)) < float(task_state.get("settle_err", 0.05)):
            return np.zeros(2, dtype=np.float32)
        return _joint_controller(q, task_state["joint_goal"], qvel=qvel, gain=0.8, damping=1.1)

    if name == "trace_circle":
        phase = task_state["phase0"] + task_state["omega"] * step_idx
        orbit_goal = target_xy + task_state["radius"] * np.array(
            [np.cos(phase), np.sin(phase)], dtype=np.float32
        )
        return _ee_controller(q, orbit_goal, qvel=qvel, gain=12.0)

    raise ValueError(f"Unknown reacher task: {name}")


def collect_reacher_multitask_episode(env, max_steps, image_size, seed, tasks):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    task_name = str(rng.choice(tasks))
    task_id = REACHER_TASKS.index(task_name)
    task_state = _init_reacher_task(env, task_name, rng)

    frames, actions, observations, rewards, dones = [], [], [], [], []
    object_to_goal_dist, tip_to_object_dist, push_contact = [], [], []
    steps = 0
    prev_action = np.zeros(2, dtype=np.float32)
    terminated = truncated = False
    while not (terminated or truncated) and steps < max_steps:
        action = _reacher_task_action(env, task_state, rng, step_idx=steps, prev_action=prev_action)
        next_obs, reward, terminated, truncated, info = env.step(action)

        frame = resize(render_env(env), size=image_size)
        frames.append(frame)
        actions.append(np.asarray(action, dtype=np.float32))
        observations.append(np.asarray(obs, dtype=np.float32))
        rewards.append(np.float32(reward))
        dones.append(bool(terminated or truncated))

        if task_name == "push_blue_object_to_blue_spot":
            obj_xy = _get_body_xy(env, "blue_object")
            goal_xy = _get_body_xy(env, "blue_goal")
            tip_xy = _get_body_xy(env, "fingertip")
            object_to_goal_dist.append(np.float32(np.linalg.norm(goal_xy - obj_xy)))
            tip_to_object_dist.append(np.float32(np.linalg.norm(tip_xy - obj_xy)))
            push_contact.append(bool(_has_geom_contact(env, "fingertip", "blue_object_geom")))
        elif task_name == "push_purple_ball_to_edge":
            obj_xy = _get_body_xy(env, "purple_ball")
            goal_xy = np.asarray(task_state["edge_goal"], dtype=np.float32)
            tip_xy = _get_body_xy(env, "fingertip")
            object_to_goal_dist.append(np.float32(np.linalg.norm(goal_xy - obj_xy)))
            tip_to_object_dist.append(np.float32(np.linalg.norm(tip_xy - obj_xy)))
            push_contact.append(bool(_has_geom_contact(env, "fingertip", "purple_ball_geom")))
        else:
            object_to_goal_dist.append(np.float32(-1.0))
            tip_to_object_dist.append(np.float32(-1.0))
            push_contact.append(False)

        obs = next_obs
        prev_action = np.asarray(action, dtype=np.float32)
        steps += 1

    episode = {
        "pixels": [f.astype(np.uint8) for f in frames],
        "action": np.stack(actions, axis=0).astype(np.float32),
        "observation": np.stack(observations, axis=0).astype(np.float32),
        "reward": np.array(rewards, dtype=np.float32),
        "done": np.array(dones, dtype=bool),
        "task_id": np.full(len(frames), task_id, dtype=np.int64),
        "task_name": [task_name for _ in range(len(frames))],
        "object_to_goal_dist": np.array(object_to_goal_dist, dtype=np.float32),
        "tip_to_object_dist": np.array(tip_to_object_dist, dtype=np.float32),
        "push_contact": np.array(push_contact, dtype=bool),
    }
    return episode, task_name


def make_env(env_name, seed, image_size=224, camera_name=None, max_episode_steps=None):
    """Create a renderable Gymnasium MuJoCo environment."""
    del camera_name
    make_kwargs = {"render_mode": "rgb_array", "width": image_size, "height": image_size}
    if max_episode_steps is not None:
        make_kwargs["max_episode_steps"] = int(max_episode_steps)
    if env_name == CUSTOM_REACHER_ENV_ID:
        env = gym.make(
            "Reacher-v5",
            xml_file=str(CUSTOM_REACHER_XML),
            disable_env_checker=True,
            **make_kwargs,
        )
    else:
        env = gym.make(env_name, **make_kwargs)
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


def _patch_is_image_column():
    import stable_worldmodel.data.formats.lance as _lance
    _orig = _lance.is_image_column
    def _patched(vals):
        if isinstance(vals, np.ndarray):
            return False
        try:
            return _orig(vals)
        except ValueError:
            return False
    _lance.is_image_column = _patched


def main():
    _patch_is_image_column()
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

    env = make_env(
        args.env,
        args.seed,
        image_size=args.image_size,
        max_episode_steps=args.max_steps,
    )
    policy = None
    if args.policy == "random":
        policy = random_policy(env)
    elif args.policy == "mixed":
        policy = mixed_random_policy(env)
    else:
        if "Reacher" not in args.env and args.env != CUSTOM_REACHER_ENV_ID:
            raise ValueError(
                "--policy reacher_multitask requires Reacher-v5 or ReacherTaskAgnostic-v0."
            )
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
