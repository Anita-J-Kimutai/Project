"""
REINFORCEMENT LEARNING-BASED QUALITY OF SERVICE-AWARE 
RESOURCE SCHEDULING IN 5G NETWORK SLICING

Description:
    Trains a Proximal Policy Optimisation (PPO) agent on
    the custom FiveGSlicingEnv simulation environment.
    The agent learns to allocate Physical Resource Blocks
    (PRBs) across eMBB, URLLC and mMTC slices to jointly
    optimise QoS and energy efficiency.

  
"""

import os
import sys
import numpy as np

# Add project root to path so imports work correctly 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project imports
from environment.env_5g import FiveGSlicingEnv

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor


# CONFIGURATION 

# Reproducibility
SEED = 42                    # Fixed seed for reproducibility

# Training parameters 

TOTAL_TIMESTEPS  = 500_000   # verified sufficient for PPO convergence in comparable small-observation-space Gymnasium environments 
LEARNING_RATE    = 3e-4      # recommended standard default for PPO
N_STEPS          = 2048      # Steps per policy update 
BATCH_SIZE       = 64        # Mini-batch size to balance between gradient accuracy and speed
N_EPOCHS         = 10        # PPO update epochs per rollout
GAMMA            = 0.99      # Discount factor
GAE_LAMBDA       = 0.95      # Generalised Advantage Estimation lambda
CLIP_RANGE       = 0.2       # PPO clipping parameter
N_ENVS           = 1         # Number of parallel environments

# Evaluation parameters 
EVAL_FREQ        = 10_000    # Evaluate every N training steps
EVAL_EPISODES    = 10        # Episodes per evaluation

# File paths 
LOG_DIR          = "./results/ppo_logs/"
MODEL_SAVE_PATH  = "./results/ppo_final_model"
BEST_MODEL_PATH  = "./results/ppo_best_model"
CHECKPOINT_PATH  = "./results/checkpoints/"


# SETUP 

def create_dirs():
    """Create output directories if they do not exist."""
    for path in [LOG_DIR, BEST_MODEL_PATH, CHECKPOINT_PATH]:
        os.makedirs(path, exist_ok=True)
    print("✅ Output directories ready.")


def make_env(seed=0):
    """Create and wrap a single environment instance."""
    env = FiveGSlicingEnv()
    env = Monitor(env, LOG_DIR)   # Logs episode rewards and lengths
    env.reset(seed=seed)
    return env


# TRAINING 

def train():
    """
    Main training function
    Creates environment, instantiates PPO agent,
    configures callbacks, and runs training
    """
    print("\n" + "="*55)
    print("  5G RL SCHEDULER — PPO TRAINING")
    print("="*55)
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Learning rate   : {LEARNING_RATE}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Random seed     : {SEED}")
    print("="*55 + "\n")

    # Create output directories 
    create_dirs()

    # Create training environment 
    print("Creating training environment...")
    train_env = make_env(seed=SEED)
    print("✅ Training environment created.\n")

    # Create evaluation environment 
    
    print("Creating evaluation environment...")
    eval_env = make_env(seed=SEED + 1)
    print("✅ Evaluation environment created.\n")

    # Instantiate PPO agent
    print("Instantiating PPO agent...")
    model = PPO(
        policy          = "MlpPolicy",   # Fully connected neural network
        env             = train_env,
        learning_rate   = LEARNING_RATE,
        n_steps         = N_STEPS,
        batch_size      = BATCH_SIZE,
        n_epochs        = N_EPOCHS,
        gamma           = GAMMA,
        gae_lambda      = GAE_LAMBDA,
        clip_range      = CLIP_RANGE,
        verbose         = 1,             # Print training progress
        seed            = SEED,
        tensorboard_log = LOG_DIR,
    )
    print("✅ PPO agent instantiated.\n")
    print(f"Policy architecture:\n{model.policy}\n")

    # Configure callbacks 
    # EvalCallback: evaluates agent periodically and saves best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = BEST_MODEL_PATH,
        log_path             = LOG_DIR,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,     # Use deterministic policy for eval
        render               = False,
    )

    # CheckpointCallback: saves model at regular intervals
    checkpoint_callback = CheckpointCallback(
        save_freq  = 50_000,
        save_path  = CHECKPOINT_PATH,
        name_prefix= "ppo_5g",
    )

    # Train the agent 
    print("Starting training...\n")
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [eval_callback, checkpoint_callback],
        progress_bar    = True,
    )

    # Save the final trained model 
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Training complete!")
    print(f"   Final model saved to : {MODEL_SAVE_PATH}.zip")
    print(f"   Best model saved to  : {BEST_MODEL_PATH}/best_model.zip")
    print(f"   Logs saved to        : {LOG_DIR}")

    # Quick evaluation of final model
    print("\nRunning quick post-training evaluation (10 episodes)...")
    evaluate_model(model, eval_env, n_episodes=10)

    train_env.close()
    eval_env.close()
    return model


# EVALUATION 

def evaluate_model(model, env, n_episodes=10):
    """
    Evaluate the trained PPO agent over N episodes.
    Prints mean reward and standard deviation.

    Args:
        model      : trained PPO model
        env        : evaluation environment
        n_episodes : number of evaluation episodes
    """
    rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Deterministic action — no random exploration
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    std_reward  = np.std(rewards)

    print(f"\n{'='*45}")
    print(f"  POST-TRAINING EVALUATION RESULTS")
    print(f"{'='*45}")
    print(f"  Episodes        : {n_episodes}")
    print(f"  Mean reward     : {mean_reward:.4f}")
    print(f"  Std deviation   : {std_reward:.4f}")
    print(f"  Min reward      : {min(rewards):.4f}")
    print(f"  Max reward      : {max(rewards):.4f}")
    print(f"{'='*45}\n")

    return mean_reward, std_reward


# ENTRY POINT 

if __name__ == "__main__":
    trained_model = train()