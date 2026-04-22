
import argparse
import os
import sys
import numpy as np
import pybullet as p
from stable_baselines3 import PPO, SAC
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.truss_assembly_env import TrussAssemblyEnv
from scripts.evaluation_utils import print_metrics, evaluate_policy_metrics
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def evaluate(args):
    print(f"Loading model from: {args.model_path}")
    print(f"Environment Stage: {args.stage}")
    
    # Create env
    raw_env = TrussAssemblyEnv(render_mode="rgb_array", curriculum_stage=args.stage, control_mode=args.control_mode)
    env = raw_env
    
    # Load model
    if "ppo" in args.model_path.lower():
        model = PPO.load(args.model_path)
    else:
        model = SAC.load(args.model_path)

    # Critical-2: Load and apply normalization stats
    stats_path = args.model_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from: {stats_path}")
        # Create a dummy vec env to wrap for normalization
        venv = DummyVecEnv([lambda: env])
        env = VecNormalize.load(stats_path, venv)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: No normalization stats found. Evaluation may be inaccurate.")
    
    n_episodes = args.episodes
    print(f"\nEvaluating for {n_episodes} episodes...")
    
    # Fix 7.1: Run evaluate_policy_metrics ONCE for accurate aggregate metrics
    metrics = evaluate_policy_metrics(model, env, n_episodes=n_episodes, deterministic=True)
    
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS (Stage {args.stage})")
    print("="*50)
    print_metrics(metrics, prefix="")
    print(f"Failures: Timeout={metrics.timeouts}, Drop={metrics.early_drops}, HighMomentumRelease={metrics.high_momentum_releases}")
    
    # Separate GIF-capture run (only the best episode)
    if args.save_gif:
        print(f"\nCapturing GIF of best episode...")
        best_episode_reward = -np.inf
        best_episode_frames = []
        for i in range(min(n_episodes, 5)):  # Limit GIF candidates to 5
            reset_res = env.reset()
            if isinstance(reset_res, tuple):
                obs, _ = reset_res
            else:
                obs = reset_res
            
            done = False
            truncated = False
            episode_reward = 0
            frames = []
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                step_res = env.step(action)
                
                # Handle VecEnv vs Gym Env return signature
                if len(step_res) == 5:
                    obs, reward, done, truncated, info = step_res
                else:
                    obs, reward, done, info = step_res
                    truncated = done # VecEnv combines them
                    info = info[0] # VecEnv returns list of dicts
                    reward = reward[0]
                    done = done[0]
                    truncated = truncated[0]
                
                episode_reward += reward
                # Use raw_env for rendering to bypass VecNormalize wrapper
                frame = raw_env.render()
                frames.append(frame)
            
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_episode_frames = frames
        
        if len(best_episode_frames) > 0:
            gif_path = os.path.join(os.path.dirname(args.model_path), "eval_best_episode.gif")
            print(f"Saving GIF to: {gif_path}")
            pil_images = [Image.fromarray(frame) for frame in best_episode_frames]
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=33,
                loop=0
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to .zip model file")
    parser.add_argument("--stage", type=int, default=5, help="Curriculum stage")
    parser.add_argument("--episodes", type=int, default=20, help="Number of test episodes")
    parser.add_argument("--save_gif", action="store_true", help="Save GIF of best episode")
    parser.add_argument("--control_mode", type=str, default="joint", choices=["joint", "task_space"], help="Control mode")
    args = parser.parse_args()
    
    evaluate(args)
