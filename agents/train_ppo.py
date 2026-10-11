"""
REINFORCEMENT LEARNING-BASED QUALITY OF SERVICE-AWARE
RESOURCE SCHEDULING IN 5G NETWORK SLICING

PPO TRAINING SCRIPT

Trains a Proximal Policy Optimisation (PPO) agent on the 
FiveGSlicingEnv. The agent learns to choose, every 0.5 ms slot,
how to split PRBs between eMBB, URLLC and mMTC and how many PRBs
to switch on, balancing QoS against base station energy use.

"""

import argparse        # reads options typed on the command line
import json            # saves the run settings as a readable file
import os
import time

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from environment.env_5g import FiveGSlicingEnv, SCENARIOS

# CONFIGURATION

DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_TIMESTEPS = 500_000      # 500k steps = 2,500 episodes of 200 slots
DEFAULT_WEIGHTS = [0.4, 0.3, 0.2, 0.1]   # w1 eMBB, w2 URLLC, w3 mMTC, w4 energy

# PPO hyperparameters 
PPO_PARAMS = {
    "learning_rate": 3e-4,   # step size for neural network updates
    "n_steps": 2048,         # slots collected before each policy update
    "batch_size": 64,        # mini-batch size for gradient updates
    "n_epochs": 10,          # passes over each batch of experience
    "gamma": 0.99,           # discount factor: how much future reward counts
    "gae_lambda": 0.95,      # Generalised Advantage Estimation smoothing
    "clip_range": 0.2,       # PPO clipping: limits how far one update can move
    "ent_coef": 0.01,        # Entropy bonus
                            
}

EVAL_FREQ = 10_000         # evaluate the agent every 10k training steps
EVAL_EPISODES = 15         # episodes per evaluation (~5 per scenario)
CHECKPOINT_FREQ = 100_000  # save a copy of the model every 100k steps

RESULTS_ROOT = "./results/v2"   # v1 results in ./results are left untouched

# HELPERS
def make_env(log_dir, seed, weights):
    """
    Create one environment for training or evaluation.

    scenario=None means a random scenario is chosen at every reset.
    Monitor records each episode's total reward and length into
    monitor.csv, which is used later to plot the learning curve.
    """
    env = FiveGSlicingEnv(scenario=None, weights=tuple(weights))
    env = Monitor(env, log_dir)
    env.reset(seed=seed)
    return env


def quick_evaluation(model, weights, seed, n_episodes=10):
    """
    runs agent on EACH scenario separately 

    deterministic=True means the agent always picks its single best
    action (no random exploration), which is how it would be deployed.
    """
    print(f"\n  {'Scenario':<16}{'Reward':>9}{'eMBB sat':>10}"
          f"{'URLLC viol':>12}{'mMTC loss':>11}{'Power W':>9}{'PRBs':>7}")

    for scenario in SCENARIOS:
        env = FiveGSlicingEnv(scenario=scenario, weights=tuple(weights))
        rewards, sat, viol, loss, power, prbs = [], [], [], [], [], []

        for ep in range(n_episodes):
            # 10_000 offset: evaluation traffic differs from training traffic
            obs, _ = env.reset(seed=seed + 10_000 + ep)
            done, ep_reward = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ep_reward += r
                sat.append(info["embb_satisfaction"])
                viol.append(info["urllc_violation"])
                loss.append(info["packet_loss"])
                power.append(info["power_watts"])
                prbs.append(info["active_prbs"])
                done = term or trunc
            rewards.append(ep_reward)

        print(f"  {scenario:<16}{np.mean(rewards):9.2f}{np.mean(sat):10.3f}"
              f"{np.mean(viol):12.3f}{np.mean(loss):11.3f}"
              f"{np.mean(power):9.1f}{np.mean(prbs):7.1f}")

# TRAINING
def train_one_seed(seed, timesteps, weights, tag):
    #Train a single PPO agent with one random seed

    # Each run gets its own folder
    run_dir = os.path.join(RESULTS_ROOT, tag, f"seed_{seed}")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  TRAINING PPO | tag={tag} | seed={seed} | steps={timesteps:,}")
    print(f"  weights (w1..w4) = {weights}")
    print("=" * 60)

    # Fix the seed for Python, NumPy and PyTorch together
    set_random_seed(seed)

    """
    Separate environments for training and evaluation. Different seeds
    mean the evaluation traffic is never identical to training traffic.
    """
    train_env = make_env(log_dir, seed, weights)
    eval_env = make_env(os.path.join(log_dir, "eval"), seed + 1, weights)

    model = PPO(
        policy="MlpPolicy",     # fully connected network (2 x 64 units)
        env=train_env,
        seed=seed,
        verbose=0,              # 0 = quiet; the progress bar shows progress
        device="cpu",           # small networks train faster on CPU than GPU
        tensorboard_log=log_dir,
        **PPO_PARAMS,           # "**" unpacks the dictionary into arguments
    )

    """
    EvalCallback: every EVAL_FREQ steps, tests the current policy and
    saves it as best_model.zip whenever it beats the previous best.
    """
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=log_dir,              # writes evaluations.npz
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        verbose=0,
    )

    """
    CheckpointCallback: saves snapshots so training progress can be
    examined later (or resumed if the computer crashes).
    """
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="ppo_5g",
    )

    start = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    minutes = (time.time() - start) / 60

    model.save(os.path.join(run_dir, "final_model"))

    # Record every setting used, so the run is fully reproducible
    config = {
        "seed": seed,
        "timesteps": timesteps,
        "weights": weights,
        "ppo_params": PPO_PARAMS,
        "eval_freq": EVAL_FREQ,
        "eval_episodes": EVAL_EPISODES,
        "training_minutes": round(minutes, 1),
        "environment_version": 2,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Training finished in {minutes:.1f} minutes")
    print(f"  Saved: {run_dir}/final_model.zip and best_model.zip")
    print("\n  Quick evaluation of best model (10 episodes per scenario):")
    best = PPO.load(os.path.join(run_dir, "best_model"), device="cpu")
    quick_evaluation(best, weights, seed)

    train_env.close()
    eval_env.close()


def parse_args():
    #Read command line options (all optional; defaults used otherwise)
    p = argparse.ArgumentParser(description="Train PPO on the 5G slicing env")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument("--weights", type=float, nargs=4, default=DEFAULT_WEIGHTS,
                   metavar=("W1", "W2", "W3", "W4"))
    p.add_argument("--tag", type=str, default="main",
                   help="name of the results subfolder, e.g. main or w4_0.3")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for s in args.seeds:
        train_one_seed(s, args.timesteps, args.weights, args.tag)
    print("\nAll training runs complete.")