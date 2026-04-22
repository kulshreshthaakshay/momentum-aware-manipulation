"""
Shared evaluation helpers for Zero-G assembly policies.

The observation layout is defined by TrussAssemblyEnv. Keeping the slices here
prevents evaluation scripts from silently diverging when the state vector changes.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


import gymnasium as gym

# 40D Observation Layout (post-refactor translation-invariant)
ROBOT_ORN = slice(0, 4)
ROBOT_LIN_VEL = slice(4, 7)
ROBOT_ANG_VEL = slice(7, 10)
EE_TO_PART = slice(24, 27)
PART_TO_GOAL = slice(27, 30)
GRIPPER_STATE = 30
CONTACT_FORCE = slice(31, 34)
DIST_TO_PART = 34
DIST_TO_GOAL = 35
H_SYS = slice(36, 39)
H_SYS_NORM = 39

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
) -> RolloutMetrics:
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
        obs, _ = env.reset()
        episode_reward = 0.0
        grasped = False
        at_goal = False
        released = False
        max_h = float(obs[H_SYS_NORM])

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            max_h = max(max_h, float(obs[H_SYS_NORM]))

            gripper_closed = bool(obs[GRIPPER_STATE] > 0.5)
            if not grasped and gripper_closed:
                grasped = True
                grasps += 1

            if grasped and not at_goal and float(obs[DIST_TO_GOAL]) < getattr(env, "at_goal_distance", 0.4):
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
                lengths.append(step + 1)
                break

            if terminated or truncated:
                if info.get("dropped_early", False):
                    early_drops += 1
                elif not info.get("success", False):
                    timeouts += 1  # Fix 7.2: count timeouts here, not in unreachable else
                lengths.append(step + 1)
                break
        else:
            # Fallback: loop exhausted without break (shouldn't happen normally)
            timeouts += 1
            lengths.append(env.max_steps)

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
