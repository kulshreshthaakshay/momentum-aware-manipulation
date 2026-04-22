"""
Continue Training from Best Model
=================================

Use this script to extend training from an existing checkpoint.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from envs.truss_assembly_env import TrussAssemblyEnv
import numpy as np


def make_env(stage, max_steps):
    def _init():
        return TrussAssemblyEnv(curriculum_stage=stage, max_steps=max_steps)
    return _init


def evaluate(model, stage, max_steps, n_episodes=50):
    """Quick evaluation."""
    env = TrussAssemblyEnv(curriculum_stage=stage, max_steps=max_steps)
    
    # Bug-8: Apply normalization for evaluation if model has stats
    if hasattr(model, "get_vec_normalize_env") and model.get_vec_normalize_env() is not None:
        stats_env = model.get_vec_normalize_env()
        eval_env = DummyVecEnv([lambda: env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
        eval_env.obs_rms = stats_env.obs_rms
        eval_env.ret_rms = stats_env.ret_rms
        eval_env.clip_obs = stats_env.clip_obs
    else:
        eval_env = DummyVecEnv([lambda: env])
        
    successes = 0
    for _ in range(n_episodes):
        obs = eval_env.reset()
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, info = eval_env.step(action)
            if info[0].get('success'):
                successes += 1
                break
            if done:
                break
    eval_env.close()
    return successes / n_episodes * 100


def continue_training(
    model_path: str,
    stage: int = 5,
    timesteps: int = 500000,
    n_envs: int = 8,
    max_steps: int = 1000,
    learning_rate: float = None,  # None = keep original
    save_dir: str = None
):
    """Continue training from an existing model."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_dir is None:
        save_dir = f"logs/continued_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("CONTINUE TRAINING FROM CHECKPOINT")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Stage: {stage}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Save Dir: {save_dir}")
    print("=" * 70)
    
    # Create environments
    train_env = SubprocVecEnv([make_env(stage, max_steps) for _ in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    # Evaluator needs Monitor for stats
    eval_env_raw = TrussAssemblyEnv(curriculum_stage=stage, max_steps=max_steps)
    eval_env = Monitor(eval_env_raw)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    
    # Bug-8: Load normalization stats
    stats_path = model_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from: {stats_path}")
        train_env = VecNormalize.load(stats_path, train_env)
        # Sync eval env stats
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
        eval_env.clip_obs = train_env.clip_obs
        
    model = PPO.load(model_path, env=train_env, device='cpu')
    
    # Optionally adjust learning rate for fine-tuning
    if learning_rate:
        print(f"Adjusting learning rate: {model.learning_rate} → {learning_rate}")
        model.learning_rate = learning_rate
        # Fix BUG-3: Update the underlying Adam optimizer
        for param_group in model.policy.optimizer.param_groups:
            param_group["lr"] = learning_rate
    
    # Initial evaluation
    initial_success = evaluate(model, stage, max_steps)
    print(f"\nInitial success rate: {initial_success:.1f}%")
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best",
        log_path=save_dir,
        eval_freq=25000,
        n_eval_episodes=20,
        deterministic=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=f"{save_dir}/checkpoints",
        name_prefix="continued"
    )
    
    # Train
    print(f"\nTraining for {timesteps:,} more timesteps...")
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=False  # Keep counting from original
    )
    
    # Final evaluation
    final_success = evaluate(model, stage, max_steps, n_episodes=100)
    print(f"\nFinal success rate: {final_success:.1f}%")
    print(f"Improvement: {final_success - initial_success:+.1f}%")
    
    # Save
    model.save(f"{save_dir}/final_model")
    print(f"\nModel saved to: {save_dir}/final_model")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue training from checkpoint")
    parser.add_argument("model_path", type=str, help="Path to model to continue from")
    parser.add_argument("--stage", type=int, default=5, help="Curriculum stage")
    parser.add_argument("--timesteps", type=int, default=500000, help="Additional timesteps")
    parser.add_argument("--n_envs", type=int, default=8, help="Parallel environments")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=None, help="New learning rate (optional)")
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory")
    
    args = parser.parse_args()
    
    model = continue_training(
        model_path=args.model_path,
        stage=args.stage,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
