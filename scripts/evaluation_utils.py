"""
Shared evaluation helpers for Zero-G assembly policies.

The observation layout is defined by TrussAssemblyEnv. Keeping the slices here
prevents evaluation scripts from silently diverging when the state vector changes.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


import gymnasium as gym

from envs.truss_assembly_env import OBS_LAYOUT

# 40D Observation Layout (post-refactor translation-invariant)
ROBOT_ORN = slice(*OBS_LAYOUT["robot_orn"])
ROBOT_LIN_VEL = slice(*OBS_LAYOUT["robot_lin_vel"])
ROBOT_ANG_VEL = slice(*OBS_LAYOUT["robot_ang_vel"])
EE_TO_PART = slice(*OBS_LAYOUT["ee_to_part"])
PART_TO_GOAL = slice(*OBS_LAYOUT["part_to_goal"])
GRIPPER_STATE = OBS_LAYOUT["gripper_state"][0]
CONTACT_FORCE = slice(*OBS_LAYOUT["contact_force"])
DIST_TO_PART = OBS_LAYOUT["dist_to_part"][0]
DIST_TO_GOAL = OBS_LAYOUT["dist_to_goal"][0]
H_SYS = slice(*OBS_LAYOUT["h_sys"])
H_SYS_NORM = OBS_LAYOUT["h_sys_norm"][0]

class MomentumMonitor(gym.Wrapper):
    """Tracks maximum momentum across an episode and logs it into info."""
    def __init__(self, env):
        super().__init__(env)
        self.max_momentum = 0.0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.max_momentum = info.get("H_sys_norm", 0.0)
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.max_momentum = max(self.max_momentum, info.get("H_sys_norm", 0.0))
        if terminated or truncated:
            info["episode_max_momentum"] = self.max_momentum
        return obs, reward, terminated, truncated, info


@dataclass
class RolloutMetrics:
    episodes: int
    successes: int
    grasps: int
    at_goals: int
    releases: int
    controlled_releases: int
    high_momentum_releases: int
    early_drops: int
    timeouts: int
    mean_reward: float
    mean_length: float
    mean_max_momentum: float
    max_momentum: float

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.episodes)

    @property
    def grasp_rate(self) -> float:
        return self.grasps / max(1, self.episodes)

    @property
    def at_goal_rate(self) -> float:
        return self.at_goals / max(1, self.episodes)

    @property
    def release_rate(self) -> float:
        return self.releases / max(1, self.episodes)

    @property
    def controlled_release_rate(self) -> float:
        return self.controlled_releases / max(1, self.episodes)

    def as_dict(self) -> Dict[str, float]:
        return {
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "grasp_rate": self.grasp_rate,
            "at_goal_rate": self.at_goal_rate,
            "release_rate": self.release_rate,
            "controlled_release_rate": self.controlled_release_rate,
            "high_momentum_release_rate": self.high_momentum_releases / max(1, self.episodes),
            "early_drop_rate": self.early_drops / max(1, self.episodes),
            "timeout_rate": self.timeouts / max(1, self.episodes),
            "mean_reward": self.mean_reward,
            "mean_length": self.mean_length,
            "mean_max_momentum": self.mean_max_momentum,
            "max_momentum": self.max_momentum,
        }


def evaluate_policy_metrics(
    model: Any,
    env: Any,
    n_episodes: int = 50,
    deterministic: bool = True,
    max_steps: int = None
) -> RolloutMetrics:
    # Significant-12: Use passed max_steps or detect from env
    if max_steps is None:
        if hasattr(env, "max_steps"):
            max_steps = env.max_steps
        elif hasattr(env, "get_attr"):
            max_steps = env.get_attr("max_steps")[0]
        else:
            max_steps = 500 # Default fallback
            
    # Significant-13: Fetch task-specific distances from env for accurate metrics
    def get_env_attr(attr_name, default):
        if hasattr(env, attr_name):
            return getattr(env, attr_name)
        elif hasattr(env, "get_attr"):
            return env.get_attr(attr_name)[0]
        return default

    grasp_dist = get_env_attr("grasp_distance", 0.25)
    goal_dist = get_env_attr("at_goal_distance", 0.40)
            
    successes = 0
    grasps = 0
    at_goals = 0
    releases = 0
    controlled_releases = 0
    high_momentum_releases = 0
    early_drops = 0
    timeouts = 0
    rewards = []
    lengths = []
    max_momenta = []

    for _ in range(n_episodes):
        reset_res = env.reset()
        if isinstance(reset_res, tuple):
            obs, _ = reset_res
        else:
            obs = reset_res
        
        # Handle batched observations from VecEnv
        is_batched = (obs.ndim > 1)
        
        episode_reward = 0.0
        grasped = False
        at_goal = False
        released = False
        
        current_obs = obs[0] if is_batched else obs
        max_h = float(current_obs[H_SYS_NORM])

        for step_idx in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            step_res = env.step(action)
            
            # Unpack step results (tuple of 5 or 4 depending on VecEnv)
            if len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
                done = terminated or truncated
                info = info
            else:
                obs, reward, done, info = step_res
                info = info[0] # VecEnv returns list of infos
                reward = reward[0]
            
            episode_reward += reward
            current_obs = obs[0] if is_batched else obs
            max_h = max(max_h, float(current_obs[H_SYS_NORM]))

            gripper_closed = bool(current_obs[GRIPPER_STATE] > 0.5)
            if not grasped and gripper_closed:
                # Check if actually holding the part
                dist_p = float(current_obs[DIST_TO_PART])
                if dist_p < grasp_dist: 
                    grasped = True
                    grasps += 1

            if grasped and not at_goal and float(current_obs[DIST_TO_GOAL]) < goal_dist:
                at_goal = True
                at_goals += 1

            if grasped and not gripper_closed and not released:
                released = True
                releases += 1

            if info.get("momentum_controlled_release", False):
                controlled_releases += 1
            if info.get("high_momentum_release", False):
                high_momentum_releases += 1

            if info.get("success", False):
                successes += 1
                lengths.append(step_idx + 1)
                break

            if done:
                if info.get("dropped_early", False):
                    early_drops += 1
                elif not info.get("success", False):
                    timeouts += 1
                lengths.append(step_idx + 1)
                break
        else:
            # Loop ended without break
            lengths.append(max_steps)
            timeouts += 1

        rewards.append(episode_reward)
        max_momenta.append(max_h)

    return RolloutMetrics(
        episodes=n_episodes,
        successes=successes,
        grasps=grasps,
        at_goals=at_goals,
        releases=releases,
        controlled_releases=controlled_releases,
        high_momentum_releases=high_momentum_releases,
        early_drops=early_drops,
        timeouts=timeouts,
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        mean_max_momentum=float(np.mean(max_momenta)) if max_momenta else 0.0,
        max_momentum=float(np.max(max_momenta)) if max_momenta else 0.0,
    )


def print_metrics(metrics: RolloutMetrics, prefix: str = "  ") -> None:
    data = metrics.as_dict()
    print(f"{prefix}Success Rate: {data['success_rate'] * 100:.1f}%")
    print(f"{prefix}Grasp Rate: {data['grasp_rate'] * 100:.1f}%")
    print(f"{prefix}At-Goal Rate: {data['at_goal_rate'] * 100:.1f}%")
    print(f"{prefix}Release Rate: {data['release_rate'] * 100:.1f}%")
    print(f"{prefix}Controlled Release Rate: {data['controlled_release_rate'] * 100:.1f}%")
    print(f"{prefix}Mean Reward: {data['mean_reward']:.1f}")
    print(f"{prefix}Mean Length: {data['mean_length']:.1f}")
    print(f"{prefix}Mean Max |H_sys|: {data['mean_max_momentum']:.4f}")
