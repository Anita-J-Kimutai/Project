"""
REINFORCEMENT LEARNING-BASED QUALITY OF SERVICE-AWARE 
RESOURCE SCHEDULING IN 5G NETWORK SLICING

COMPREHENSIVE EVALUATION FRAMEWORK

Description:
    Comparatively evaluates the trained PPO agent against
    three traditional baseline schedulers across three
    defined traffic scenarios.

    Schedulers evaluated:
        - PPO (trained Reinforment Learning agent)
        - Round Robin
        - Proportional Fair
        - Max-Throughput

    Traffic scenarios:
        - Balanced    : equal traffic across all slices
        - eMBB-dom    : high eMBB video streaming demand
        - URLLC-crit  : spike in latency-sensitive traffic

    Metrics collected per scheduler per scenario:
        - Mean throughput (Mbps)
        - Mean latency (ms)
        - Mean packet loss rate
        - Jain's Fairness Index
        - Mean energy consumption
        - Mean episode reward

    Results saved to: ./results/evaluation_results.csv
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from environment.env_5g import FiveGSlicingEnv
from baselines.schedulers import (
    RoundRobinScheduler,
    ProportionalFairScheduler,
    MaxThroughputScheduler,
)
from stable_baselines3 import PPO


# CONFIGURATION

N_EPISODES       = 100     # Episodes per scheduler per scenario
SEED             = 42      # Fixed seed for reproducibility
MODEL_PATH       = "./results/ppo_final_model"
RESULTS_PATH     = "./results/evaluation_results.csv"

"""
Traffic scenario configurations 

Each scenario adjusts the Poisson lambda values in env
by modifying the env parameters after reset
"""
SCENARIOS = {
    "Balanced": {
        "description": "Equal traffic across all three slices",
        "embb_lambda" : 50,   # Standard eMBB demand
        "urllc_lambda": 10,   # Standard URLLC demand
        "mmtc_lambda" : 80,   # Standard mMTC demand
    },
    "eMBB-Dominant": {
        "description": "High video streaming demand on eMBB slice",
        "embb_lambda" : 90,   # High eMBB demand
        "urllc_lambda": 5,    # Low URLLC demand
        "mmtc_lambda" : 40,   # Low mMTC demand
    },
    "URLLC-Critical": {
        "description": "Spike in latency-sensitive URLLC traffic",
        "embb_lambda" : 20,   # Low eMBB demand
        "urllc_lambda": 40,   # High URLLC demand
        "mmtc_lambda" : 60,   # Moderate mMTC demand
    },
}


# JAIN'S FAIRNESS INDEX

def jains_fairness_index(values):
    """
    Compute Jain's Fairness Index for a list of values.
    JFI = (sum(x))^2 / (n * sum(x^2))
    Range: [1/n, 1] where 1 = perfectly fair.

    Args:
        values (list): per-slice metric values

    Returns:
        float: Jain's Fairness Index
    """
    values = np.array(values)
    n = len(values)
    if n == 0 or np.sum(values ** 2) == 0:
        return 0.0
    return (np.sum(values) ** 2) / (n * np.sum(values ** 2))


# PPO EVALUATION

def evaluate_ppo(model, env, scenario_config,
                 n_episodes=100, seed=42):
    """
    Evaluate the trained PPO agent over N episodes.

    Args:
        model          : trained PPO model
        env            : FiveGSlicingEnv instance
        scenario_config: dict with lambda values for scenario
        n_episodes     : number of evaluation episodes
        seed           : random seed

    Returns:
        dict of mean metrics across all episodes
    """

    # Apply scenario traffic parameters
    env.embb_lambda  = scenario_config["embb_lambda"]
    env.urllc_lambda = scenario_config["urllc_lambda"]
    env.mmtc_lambda  = scenario_config["mmtc_lambda"]

    all_rewards      = []
    all_throughputs  = []
    all_latencies    = []
    all_packet_losses= []
    all_energies     = []
    all_fairness     = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        ep_reward      = 0
        ep_throughput  = []
        ep_latency     = []
        ep_packet_loss = []
        ep_energy      = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = \
                env.step(action)

            ep_reward      += reward
            ep_throughput  .append(info.get("embb_throughput", 0))
            ep_latency     .append(info.get("urllc_latency",   0))
            ep_packet_loss .append(info.get("packet_loss",     0))
            ep_energy      .append(info.get("energy",          0))
            done = terminated or truncated

        # Jain's Fairness across slice metrics
        fairness = jains_fairness_index([
            np.mean(ep_throughput),
            1 - np.mean(ep_latency) / 10,
            1 - np.mean(ep_packet_loss),
        ])

        all_rewards      .append(ep_reward)
        all_throughputs  .append(np.mean(ep_throughput))
        all_latencies    .append(np.mean(ep_latency))
        all_packet_losses.append(np.mean(ep_packet_loss))
        all_energies     .append(np.mean(ep_energy))
        all_fairness     .append(fairness)

    return {
        "mean_reward"     : np.mean(all_rewards),
        "std_reward"      : np.std(all_rewards),
        "mean_throughput" : np.mean(all_throughputs),
        "std_throughput"  : np.std(all_throughputs),
        "mean_latency"    : np.mean(all_latencies),
        "std_latency"     : np.std(all_latencies),
        "mean_packet_loss": np.mean(all_packet_losses),
        "std_packet_loss" : np.std(all_packet_losses),
        "mean_energy"     : np.mean(all_energies),
        "std_energy"      : np.std(all_energies),
        "mean_fairness"   : np.mean(all_fairness),
        "std_fairness"    : np.std(all_fairness),
    }


# BASELINE EVALUATION

def evaluate_baseline(scheduler, env, scenario_config,
                       n_episodes=100, seed=42):
    """
    Evaluate a baseline scheduler over N episodes.

    Args:
        scheduler      : BaseScheduler instance
        env            : FiveGSlicingEnv instance
        scenario_config: dict with lambda values for scenario
        n_episodes     : number of evaluation episodes
        seed           : random seed

    Returns:
        dict of mean metrics across all episodes
    """
    # Apply scenario traffic parameters
    env.embb_lambda  = scenario_config["embb_lambda"]
    env.urllc_lambda = scenario_config["urllc_lambda"]
    env.mmtc_lambda  = scenario_config["mmtc_lambda"]

    all_rewards      = []
    all_throughputs  = []
    all_latencies    = []
    all_packet_losses= []
    all_energies     = []
    all_fairness     = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        scheduler.reset()
        ep_reward      = 0
        ep_throughput  = []
        ep_latency     = []
        ep_packet_loss = []
        ep_energy      = []
        done = False

        while not done:
            action = scheduler.allocate(obs)
            obs, reward, terminated, truncated, info = \
                env.step(action)

            ep_reward      += reward
            ep_throughput  .append(info.get("embb_throughput", 0))
            ep_latency     .append(info.get("urllc_latency",   0))
            ep_packet_loss .append(info.get("packet_loss",     0))
            ep_energy      .append(info.get("energy",          0))
            done = terminated or truncated

        fairness = jains_fairness_index([
            np.mean(ep_throughput),
            1 - np.mean(ep_latency) / 10,
            1 - np.mean(ep_packet_loss),
        ])

        all_rewards      .append(ep_reward)
        all_throughputs  .append(np.mean(ep_throughput))
        all_latencies    .append(np.mean(ep_latency))
        all_packet_losses.append(np.mean(ep_packet_loss))
        all_energies     .append(np.mean(ep_energy))
        all_fairness     .append(fairness)

    return {
        "mean_reward"     : np.mean(all_rewards),
        "std_reward"      : np.std(all_rewards),
        "mean_throughput" : np.mean(all_throughputs),
        "std_throughput"  : np.std(all_throughputs),
        "mean_latency"    : np.mean(all_latencies),
        "std_latency"     : np.std(all_latencies),
        "mean_packet_loss": np.mean(all_packet_losses),
        "std_packet_loss" : np.std(all_packet_losses),
        "mean_energy"     : np.mean(all_energies),
        "std_energy"      : np.std(all_energies),
        "mean_fairness"   : np.mean(all_fairness),
        "std_fairness"    : np.std(all_fairness),
    }


# MAIN EVALUATION LOOP 

def run_full_evaluation():
    """
    Run complete evaluation — all schedulers, all scenarios.
    Saves results to CSV for dashboard visualisation
    """
    print("\n" + "="*60)
    print("  5G RL SCHEDULER — COMPREHENSIVE EVALUATION")
    print("="*60)
    print(f"  Episodes per scenario : {N_EPISODES}")
    print(f"  Scenarios             : {len(SCENARIOS)}")
    print(f"  Schedulers            : 4 (PPO + 3 baselines)")
    print(f"  Random seed           : {SEED}")
    print("="*60)

    # Load environment
    print("\nLoading environment...")
    env = FiveGSlicingEnv()
    print("✅ Environment loaded")

    # Load trained PPO model 
    print(f"\nLoading PPO model from {MODEL_PATH}...")
    ppo_model = PPO.load(MODEL_PATH, env=env)
    print("✅ PPO model loaded.")

    # Define schedulers
    schedulers = {
        "PPO"               : None,   # separately handled
        "Round Robin"       : RoundRobinScheduler(),
        "Proportional Fair" : ProportionalFairScheduler(),
        "Max-Throughput"    : MaxThroughputScheduler(),
    }

    # Results storage
    results = []

    # Run evaluation loops
    for scenario_name, scenario_config in SCENARIOS.items():
        print(f"\n{'─'*60}")
        print(f"  Scenario: {scenario_name}")
        print(f"  {scenario_config['description']}")
        print(f"{'─'*60}")

        for scheduler_name, scheduler in schedulers.items():
            print(f"\n  Evaluating {scheduler_name}...", end=" ")

            if scheduler_name == "PPO":
                metrics = evaluate_ppo(
                    ppo_model, env, scenario_config,
                    n_episodes=N_EPISODES, seed=SEED
                )
            else:
                metrics = evaluate_baseline(
                    scheduler, env, scenario_config,
                    n_episodes=N_EPISODES, seed=SEED
                )

            # Store result row
            row = {
                "scenario"        : scenario_name,
                "scheduler"       : scheduler_name,
                **metrics
            }
            results.append(row)

            print(f"reward={metrics['mean_reward']:.3f} | "
                  f"throughput={metrics['mean_throughput']:.2f} | "
                  f"latency={metrics['mean_latency']:.2f} | "
                  f"energy={metrics['mean_energy']:.2f}")

    # Save results to CSV
    os.makedirs("./results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n\n✅ Results saved to: {RESULTS_PATH}")

    # Print summary table
    print("\n" + "="*60)
    print("  EVALUATION SUMMARY")
    print("="*60)

    for scenario_name in SCENARIOS:
        print(f"\n  [{scenario_name}]")
        print(f"  {'Scheduler':<20} {'Reward':>8} "
              f"{'Throughput':>12} {'Latency':>9} "
              f"{'Energy':>8} {'Fairness':>10}")
        print(f"  {'-'*70}")

        scenario_results = [
            r for r in results
            if r["scenario"] == scenario_name
        ]

        for r in scenario_results:
            print(
                f"  {r['scheduler']:<20} "
                f"{r['mean_reward']:>8.3f} "
                f"{r['mean_throughput']:>12.3f} "
                f"{r['mean_latency']:>9.3f} "
                f"{r['mean_energy']:>8.3f} "
                f"{r['mean_fairness']:>10.4f}"
            )

    print("\n✅ Evaluation complete!")
    env.close()
    return df


# ENTRY POINT

if __name__ == "__main__":
    df = run_full_evaluation()