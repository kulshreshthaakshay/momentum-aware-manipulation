"""
Curriculum Training for Zero-G Truss Assembly
==============================================

This script implements progressive curriculum learning through stages 1-5,
building skills incrementally:

Stage 1: Station keeping (maintain position)
Stage 2: Approach (navigate to part)
Stage 3: Grasp (close gripper at right time)
Stage 4: Transport (move part to goal)
Stage 5: Full assembly (approach → grasp → transport → release)

Uses the optimized hyperparameters identified through tuning.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from envs.truss_assembly_env import TrussAssemblyEnv
from scripts.evaluation_utils import evaluate_policy_metrics
import json


# Stage-specific configurations
# Fix 4.1: Practical success thresholds (100% is unreachable with stochastic eval)
STAGE_CONFIGS = {
    1: {
        "name": "Station Keeping",
        "max_steps": 500,
        "success_threshold": 0.95,
        "min_timesteps": 50000,
        "max_timesteps": 200000,
    },
    2: {
        "name": "Approach",
        "max_steps": 500,
        "success_threshold": 0.90,
        "min_timesteps": 50000,
        "max_timesteps": 200000,
    },
    3: {
        "name": "Grasp",
        "max_steps": 600,
        "success_threshold": 0.85,
        "min_timesteps": 150000,
        "max_timesteps": 500000,
    },
    4: {
        "name": "Transport",
        "max_steps": 800,
        "success_threshold": 0.80,
        "min_timesteps": 100000,
        "max_timesteps": 400000,
    },
    5: {
        "name": "Full Assembly",
        "max_steps": 1000,
        "success_threshold": 0.75,
        "min_timesteps": 200000,
        "max_timesteps": 1000000,
    },
}

# Optimized hyperparameters (from tuning results)
PPO_CONFIG = {
    "learning_rate": 2e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 15,
    "gamma": 0.995,
    "gae_lambda": 0.97,
    "ent_coef": 0.03,
    "clip_range": 0.3,
    "vf_coef": 0.25,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [512, 256, 128]}
}


class SuccessRateCallback(BaseCallback):
    """Track success rate during training."""
    
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.successes = 0
        self.episodes = 0
        self.success_rates = []
        
    def _on_step(self):
        # Fix 4.3: Only count at episode boundaries, single condition to avoid double-counting
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episodes += 1
                    if info.get('success', False):
                        self.successes += 1
        
        if self.n_calls % self.check_freq == 0 and self.episodes > 0:
            rate = self.successes / self.episodes
            self.success_rates.append(rate)
            if self.verbose:
                print(f"  Training success rate: {rate*100:.1f}% ({self.successes}/{self.episodes})")
            # Reset counters
            self.successes = 0
            self.episodes = 0
        
        return True
    
    def get_recent_success_rate(self):
        if len(self.success_rates) >= 3:
            return np.mean(self.success_rates[-3:])
        elif self.success_rates:
            return self.success_rates[-1]
        return 0.0


def make_env(stage, max_steps, control_mode="joint"):
    """Create environment factory."""
    from scripts.evaluation_utils import MomentumMonitor
    def _init():
        env = TrussAssemblyEnv(curriculum_stage=stage, max_steps=max_steps, control_mode=control_mode)
        env = MomentumMonitor(env)
        return Monitor(
            env,
            info_keywords=(
                "success",
                "dropped_early",
                "H_sys_norm",
                "episode_max_momentum",
                "momentum_controlled_release",
                "high_momentum_release",
            ),
        )
    return _init


def evaluate_stage(model, stage, max_steps, control_mode="joint", n_episodes=50):
    """Evaluate model on a specific stage."""
    env = TrussAssemblyEnv(curriculum_stage=stage, max_steps=max_steps, control_mode=control_mode)
    metrics = evaluate_policy_metrics(model, env, n_episodes=n_episodes, deterministic=True)
    env.close()
    return metrics


def run_curriculum(
    start_stage=1,
    n_envs=8,
    save_dir=None,
    continue_from=None,
    verbose=True,
    control_mode="joint"
):
    """Run full curriculum training from stage 1 to 5."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_dir is None:
        save_dir = f"logs/curriculum_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("CURRICULUM TRAINING: Zero-G Truss Assembly")
    print("=" * 70)
    print(f"Stages: {start_stage} → 5")
    print(f"Parallel Environments: {n_envs}")
    print(f"Save Directory: {save_dir}")
    print("=" * 70)
    
    curriculum_log = {
        "start_time": datetime.now().isoformat(),
        "stages": {}
    }
    
    model = None
    
    for stage in range(start_stage, 6):
        config = STAGE_CONFIGS[stage]
        
        print(f"\n{'=' * 70}")
        print(f"STAGE {stage}: {config['name']}")
        print(f"{'=' * 70}")
        print(f"  Max Steps: {config['max_steps']}")
        print(f"  Success Threshold: {config['success_threshold']*100:.0f}%")
        print(f"  Min Timesteps: {config['min_timesteps']:,}")
        print(f"  Max Timesteps: {config['max_timesteps']:,}")
        
        # Create environments for this stage
        train_env = SubprocVecEnv([make_env(stage, config['max_steps'], control_mode) for _ in range(n_envs)])
        eval_env = make_env(stage, config['max_steps'], control_mode)()
        
        # Create or transfer model
        if model is None:
            if continue_from:
                print(f"\n  Loading model from: {continue_from}")
                model = PPO.load(continue_from, env=train_env, device='cpu')
            else:
                print("\n  Creating new model...")
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    verbose=0,
                    tensorboard_log=f"{save_dir}/tensorboard",
                    device='cpu',
                    **PPO_CONFIG
                )
        else:
            # Transfer to new environment
            print("\n  Transferring model to new stage...")
            model.set_env(train_env)
            # Fix 4.2: Reduce LR AND force optimizer update for later stages
            if stage >= 4:
                new_lr = PPO_CONFIG["learning_rate"] / 2
                model.learning_rate = new_lr
                for param_group in model.policy.optimizer.param_groups:
                    param_group['lr'] = new_lr
        
        # Fix 4.5: Validate buffer size is divisible by batch_size
        assert (n_envs * PPO_CONFIG["n_steps"]) % PPO_CONFIG["batch_size"] == 0, \
            f"n_envs={n_envs} * n_steps={PPO_CONFIG['n_steps']} not divisible by batch_size={PPO_CONFIG['batch_size']}"
        
        # Callbacks
        success_callback = SuccessRateCallback(check_freq=25000, verbose=verbose)
        # Fix 4.4: Use deterministic=False for early stages (stochastic exploration)
        eval_deterministic = stage >= 4
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{save_dir}/stage{stage}_best",
            log_path=f"{save_dir}/stage{stage}_logs",
            eval_freq=max(config['min_timesteps'] // 4, 10000),
            n_eval_episodes=20,
            deterministic=eval_deterministic,
            verbose=1
        )
        
        # Train this stage
        print(f"\n  Training Stage {stage}...")
        
        stage_start_time = datetime.now()
        total_trained = 0
        best_success_rate = 0.0
        consecutive_successes = 0
        
        while total_trained < config['max_timesteps']:
            # Train in chunks
            chunk_size = min(config['min_timesteps'], config['max_timesteps'] - total_trained)
            
            model.learn(
                total_timesteps=chunk_size,
                callback=[success_callback, eval_callback],
                progress_bar=True,
                reset_num_timesteps=(total_trained == 0)
            )
            
            total_trained += chunk_size
            
            # Evaluate
            metrics = evaluate_stage(model, stage, config['max_steps'], control_mode, n_episodes=50)
            success_rate = metrics.success_rate
            print(f"\n  Evaluation: {success_rate*100:.1f}% success ({total_trained:,} timesteps)")
            print(f"  Mean max |H_sys|: {metrics.mean_max_momentum:.4f}")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                model.save(f"{save_dir}/stage{stage}_best_model")
            
            # Check if threshold reached
            if success_rate >= config['success_threshold']:
                consecutive_successes += 1
                print(f"  Consecutive evaluations meeting threshold: {consecutive_successes}/2")
            else:
                consecutive_successes = 0
                
            if consecutive_successes >= 2 and total_trained >= config['min_timesteps']:
                print(f"\n  ✅ Stage {stage} COMPLETE! Success rate: {success_rate*100:.1f}%")
                break
        else:
            print(f"\n  ⚠️ Stage {stage} max timesteps reached. Best: {best_success_rate*100:.1f}%")
        
        stage_duration = (datetime.now() - stage_start_time).total_seconds()
        
        # Save stage model
        model.save(f"{save_dir}/stage{stage}_final")
        
        # Log stage results
        curriculum_log["stages"][f"stage_{stage}"] = {
            "name": config['name'],
            "timesteps_trained": total_trained,
            "best_success_rate": float(best_success_rate),
            "final_success_rate": float(success_rate),
            "threshold": config['success_threshold'],
            "threshold_reached": success_rate >= config['success_threshold'],
            "mean_max_momentum": float(metrics.mean_max_momentum),
            "controlled_release_rate": float(metrics.controlled_release_rate),
            "duration_seconds": stage_duration
        }
        
        # Cleanup
        train_env.close()
        eval_env.close()
        
        # Save progress
        with open(f"{save_dir}/curriculum_log.json", 'w') as f:
            json.dump(curriculum_log, f, indent=2)
    
    # Final summary
    print("\n" + "=" * 70)
    print("CURRICULUM TRAINING COMPLETE")
    print("=" * 70)
    
    total_time = 0
    for stage_key, stage_data in curriculum_log["stages"].items():
        stage_num = stage_key.split("_")[1]
        status = "✅" if stage_data["threshold_reached"] else "⚠️"
        print(f"  {status} Stage {stage_num}: {stage_data['final_success_rate']*100:.1f}% "
              f"(target: {stage_data['threshold']*100:.0f}%)")
        total_time += stage_data["duration_seconds"]
    
    print(f"\n  Total Training Time: {total_time/60:.1f} minutes")
    print(f"  Models saved to: {save_dir}/")
    
    curriculum_log["end_time"] = datetime.now().isoformat()
    curriculum_log["total_duration_seconds"] = total_time
    
    with open(f"{save_dir}/curriculum_log.json", 'w') as f:
        json.dump(curriculum_log, f, indent=2)
    
    return model, curriculum_log


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Curriculum Training for Zero-G Truss Assembly")
    parser.add_argument("--start_stage", type=int, default=1,
                        help="Stage to start from (1-5)")
    parser.add_argument("--n_envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save models")
    parser.add_argument("--continue_from", type=str, default=None,
                        help="Path to model to continue from")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with reduced timesteps")
    
    parser.add_argument("--control_mode", type=str, default="joint", choices=["joint", "task_space"],
                        help="Control mode (joint or task_space)")
    
    args = parser.parse_args()
    
    if args.quick:
        print("Quick mode: Reduced timesteps for testing")
        for stage in STAGE_CONFIGS:
            STAGE_CONFIGS[stage]["min_timesteps"] //= 4
            STAGE_CONFIGS[stage]["max_timesteps"] //= 4
    
    model, log = run_curriculum(
        start_stage=args.start_stage,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        continue_from=args.continue_from,
        control_mode=args.control_mode
    )
    
    print("\n🎉 Curriculum training finished!")
    print(f"Final Stage 5 success rate: {log['stages'].get('stage_5', {}).get('final_success_rate', 0)*100:.1f}%")
