"""Reacher-v5 with custom XML + reach/push evaluation support."""
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CUSTOM_REACHER_XML = str(SCRIPT_DIR.parent / "assets" / "reacher_task_agnostic.xml")
CUSTOM_ENV_ID = "swm/ReacherCustom-v0"

N_ARM_JOINTS = 2  # first 2 of 10 qpos dims are the arm joints

# qpos layout: [arm0, arm1, target_x, target_y, blue_goal_x, blue_goal_y, blue_obj_x, blue_obj_y, purple_x, purple_y]
_IDX_TARGET = slice(2, 4)
_IDX_BLUE_GOAL = slice(4, 6)
_IDX_BLUE_OBJ = slice(6, 8)

QPOS_THRESHOLD = 0.05
FINGERTIP_DIST_THRESHOLD = 0.04
PUSH_DIST_THRESHOLD = 0.025


class ReacherCustomEvalEnv(gym.Env):
    """Reacher-v5 with custom task-agnostic XML.

    Supports three task modes via the ``task`` init param:

    ==============  =========================================================
    Mode            Success condition
    ==============  =========================================================
    ``qpos_match``  Arm joint angles match a stored target qpos (first 2 dims)
    ``reach``       Fingertip position matches target body position
    ``push``        Blue-object position matches blue-goal position
    ==============  =========================================================

    Callables used by eval:
      - ``set_state(qpos, qvel)`` – set full MuJoCo state.
      - ``set_target_qpos(goal_qpos)`` – set the goal from a future qpos vector;
        the env extracts the relevant sub-vector for the current task mode.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, xml_file=None, task="qpos_match", render_mode="rgb_array", width=224, height=224, **kwargs):
        if xml_file is None:
            xml_file = CUSTOM_REACHER_XML
        from gymnasium.envs.mujoco.reacher_v5 import ReacherEnv
        self._env = ReacherEnv(xml_file=str(xml_file), render_mode=render_mode, width=width, height=height)
        self.task = task
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.spec = gym.spec(CUSTOM_ENV_ID)
        self._clear_goal()

    def _clear_goal(self):
        self.target_qpos = None
        self.target_xy = None
        self.goal_xy = None

    def set_state(self, qpos, qvel):
        self._env.data.qpos[:] = np.copy(np.asarray(qpos).ravel())
        self._env.data.qvel[:] = np.copy(np.asarray(qvel).ravel())
        if self._env.model.na == 0:
            self._env.data.act[:] = None
        import mujoco
        mujoco.mj_forward(self._env.model, self._env.data)

    def set_target_qpos(self, target_qpos):
        target_qpos = np.asarray(target_qpos).ravel()
        if self.task == "qpos_match":
            self.target_qpos = target_qpos[:N_ARM_JOINTS].copy()
        elif self.task == "reach":
            self.target_xy = target_qpos[_IDX_TARGET].copy()
        elif self.task == "push":
            self.goal_xy = target_qpos[_IDX_BLUE_GOAL].copy()
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)

        if terminated:
            return obs, reward, terminated, truncated, info

        if self.task == "qpos_match" and self.target_qpos is not None:
            diff = np.abs(self._env.data.qpos[:N_ARM_JOINTS] - self.target_qpos)
            if np.all(diff < QPOS_THRESHOLD):
                terminated = True

        elif self.task == "reach" and self.target_xy is not None:
            tip = self._env.data.body("fingertip").xpos[:2]
            if np.linalg.norm(tip - self.target_xy) < FINGERTIP_DIST_THRESHOLD:
                terminated = True

        elif self.task == "push" and self.goal_xy is not None:
            obj = self._env.data.body("blue_object").xpos[:2]
            if np.linalg.norm(obj - self.goal_xy) < PUSH_DIST_THRESHOLD:
                terminated = True

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self._clear_goal()
        return self._env.reset(seed=seed, options=options)

    def render(self):
        return self._env.render()

    @property
    def data(self):
        return self._env.data

    def close(self):
        self._env.close()


gym.register(
    id=CUSTOM_ENV_ID,
    entry_point="scripts.reacher_custom_env:ReacherCustomEvalEnv",
)
