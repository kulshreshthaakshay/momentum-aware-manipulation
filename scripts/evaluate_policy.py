
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
from scripts.evaluation_utils import H_SYS_NORM, print_metrics, evaluate_policy_metrics

def evaluate(args):
    print(f"Loading model from: {args.model_path}")
    print(f"Environment Stage: {args.stage}")
    
    # Create env
    env = TrussAssemblyEnv(render_mode="rgb_array", curriculum_stage=args.stage, control_mode=args.control_mode)
    
    # Load model
    if "ppo" in args.model_path.lower():
        model = PPO.load(args.model_path)
    else:
        model = SAC.load(args.model_path)
    
    n_episodes = args.episodes
    best_episode_reward = -np.inf
    best_episode_frames = []
    
    print(f"\nEvaluating for {n_episodes} episodes...")
    
    for i in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        frames = []
        max_momentum = float(obs[H_SYS_NORM])
        
        step = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            max_momentum = max(max_momentum, float(obs[H_SYS_NORM]))
            step += 1
            
            # Capture frame for best episode
            if args.save_gif:
                frame = env.render()
                frames.append(frame)
        
        if info.get("success", False):
            print(f"  Episode {i+1}: SUCCESS (Reward: {episode_reward:.1f}, Max |H|: {max_momentum:.4f})")
        else:
            reason = "Timeout"
            if info.get("dropped_early", False):
                reason = "Dropped Early"
            print(f"  Episode {i+1}: FAIL - {reason} (Reward: {episode_reward:.1f}, Max |H|: {max_momentum:.4f})")
        
        # Save best episode GIF
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            best_episode_frames = frames
            
    metrics = evaluate_policy_metrics(model, env, n_episodes=n_episodes, deterministic=True)
    
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS (Stage {args.stage})")
    print("="*50)
    print_metrics(metrics, prefix="")
    print(f"Failures: Timeout={metrics.timeouts}, Drop={metrics.early_drops}, HighMomentumRelease={metrics.high_momentum_releases}")
    
    if args.save_gif and len(best_episode_frames) > 0:
        gif_path = os.path.join(os.path.dirname(args.model_path), "eval_best_episode.gif")
        print(f"\nSaving GIF to: {gif_path}")
        
        # Convert frames to PIL images
        pil_images = [Image.fromarray(frame) for frame in best_episode_frames]
        pil_images[0].save(
            gif_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=33,  # 30 fps
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
