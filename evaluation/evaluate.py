"""
REINFORCEMENT LEARNING-BASED QUALITY OF SERVICE-AWARE
RESOURCE SCHEDULING IN 5G NETWORK SLICING

Compares the trained PPO agent against four baseline schedulers on
all three traffic scenarios, using identical traffic for every
scheduler so the comparison is fair.

    Schedulers : PPO (seeds 42, 43, 44), Round Robin, Proportional Fair,
                 Max-Throughput, Best Static
    Scenarios  : Balanced, eMBB-Dominant, URLLC-Critical
    Episodes   : 100 per scheduler per scenario (per PPO seed)

Outputs 
    evaluation_results.csv   - summary: mean, std, 95% CI per scheduler/scenario
    evaluation_episodes.csv  - one row per episode (for box plots, dashboard)
    ppo_seed_summary.csv     - PPO results for each seed separately
    significance_tests.csv   - PPO vs best classical and Best Static t-tests
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from environment.env_5g import FiveGSlicingEnv, SCENARIOS, ACTION_MAP
from baselines.schedulers import (
    RoundRobinScheduler, ProportionalFairScheduler,
    MaxThroughputScheduler, StaticScheduler, find_best_static,
)

# CONFIGURATION
RESULTS_DIR = "./results/v2"
MODEL_DIR = os.path.join(RESULTS_DIR, "main")
PPO_SEEDS = [42, 43, 44]
N_EPISODES = 100

# Evaluation traffic uses seeds 20000, 20001, ... These are far away
# from the seeds used in training (42-44), Best Static search (1000+)
# and the training-time evaluation, so no scheduler is tested on
# traffic it has already seen.
EVAL_SEED = 20_000

WEIGHTS = (0.4, 0.3, 0.2, 0.1)

# Columns reported for every scheduler (see run_episode below)
METRICS = [
    "reward", "embb_throughput", "embb_satisfaction",
    "urllc_latency", "urllc_violation_rate", "mmtc_packet_loss",
    "power_watts", "active_prbs", "energy_joules",
    "energy_efficiency", "jain_fairness",
    "embb_prbs", "urllc_prbs", "mmtc_prbs",
]

# SCHEDULER WRAPPER FOR PPO
class PPOScheduler:
    """
    Wraps a trained SB3 model so it has the same allocate()/reset()
    interface as the baseline schedulers. This means one function
    (run_episode) can evaluate every scheduler in exactly the same way.
    """

    def __init__(self, model_path, name):
        self.name = name
        self.model = PPO.load(model_path, device="cpu")

    def allocate(self, obs):
        # deterministic=True: always take the best action (no exploration)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def reset(self):
        pass   # PPO keeps no memory between episodes

# METRIC HELPERS
def jain_fairness(values):
    """
    Jain's Fairness Index (Jain et al., 1984):

        J = (sum of x)^2 / (n * sum of x^2)

    J = 1 when every slice gets the same QoS score; J = 1/n (0.33 for
    three slices) when one slice gets everything. All x must be on the
    same scale, so per-slice QoS scores in [0, 1] are used:
        eMBB  - satisfaction (share of demand served)
        URLLC - 1 - SLA violation (1 if the 1 ms target was met)
        mMTC  - 1 - packet loss rate
    """
    x = np.asarray(values, dtype=float)
    if np.sum(x ** 2) == 0:
        return 1.0                     # all zero = equally (un)served
    return float(np.sum(x) ** 2 / (len(x) * np.sum(x ** 2)))


def confidence_interval_95(data):
    """
    Half-width of a 95% confidence interval for the mean:
        1.96 * standard deviation / sqrt(n)
    (normal approximation; accurate for n >= 30, and n is 100 or 300 here)
    Report results as  mean +/- this value.
    """
    data = np.asarray(data, dtype=float)
    if len(data) < 2:
        return 0.0
    return float(1.96 * np.std(data, ddof=1) / math.sqrt(len(data)))


def welch_t_test(a, b):
    """
    Welch's t-test: are the means of a and b significantly different?
    Does not assume equal variances. Returns (t statistic, p value).
    The p value uses the normal approximation, which is accurate for
    the large samples used here (100+ episodes), so SciPy is not needed.
    p < 0.05 -> the difference is statistically significant.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    if se == 0:
        return 0.0, 1.0
    t = (np.mean(a) - np.mean(b)) / se
    p = math.erfc(abs(t) / math.sqrt(2))    # two-sided p value
    return float(t), float(p)

# RUN ONE EPISODE
def run_episode(scheduler, env, seed):
    """
    Run one 200-slot episode and return its per-episode metrics.
    Every scheduler gets the same seed for the same episode number,
    so they all face exactly the same traffic.
    """
    obs, _ = env.reset(seed=seed)
    scheduler.reset()

    rewards, thr, sat, lat, viol, loss = [], [], [], [], [], []
    power, prbs, joules, fairness = [], [], [], []
    slice_prbs = []        # [eMBB, URLLC, mMTC] PRBs each slot (for the heatmap)

    done = False
    while not done:
        action = scheduler.allocate(obs)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc

        rewards.append(r)
        thr.append(info["embb_throughput"])
        sat.append(info["embb_satisfaction"])
        lat.append(info["urllc_latency"])
        viol.append(info["urllc_violation"])
        loss.append(info["packet_loss"])
        power.append(info["power_watts"])
        prbs.append(info["active_prbs"])
        joules.append(info["energy_joules"])
        slice_prbs.append([info["embb_prbs"], info["urllc_prbs"], info["mmtc_prbs"]])
        fairness.append(jain_fairness([
            info["embb_satisfaction"],
            1.0 - info["urllc_violation"],
            1.0 - info["packet_loss"],
        ]))

    # Energy efficiency = data delivered / energy used (Mbit per Joule).
    # Data per slot (Mbit) = throughput (Mbps) x slot duration (s).
    slot_s = env.slot_ms / 1000.0
    total_mbit = np.sum(thr) * slot_s
    total_joules = np.sum(joules)
    mean_slice = np.mean(slice_prbs, axis=0)    # average PRBs per slice

    return {
        "reward": np.sum(rewards),                 # total reward per episode
        "embb_throughput": np.mean(thr),           # Mbps
        "embb_satisfaction": np.mean(sat),         # 0-1
        "urllc_latency": np.mean(lat),             # ms
        "urllc_violation_rate": np.mean(viol),     # share of slots > 1 ms
        "mmtc_packet_loss": np.mean(loss),         # 0-1
        "power_watts": np.mean(power),             # W
        "active_prbs": np.mean(prbs),
        "energy_joules": total_joules,             # J per episode (100 ms)
        "energy_efficiency": total_mbit / total_joules,   # Mbit/J
        "jain_fairness": np.mean(fairness),        # 0.33-1
        "embb_prbs": mean_slice[0],                # mean PRBs given to each slice
        "urllc_prbs": mean_slice[1],
        "mmtc_prbs": mean_slice[2],
    }


def evaluate_scheduler(scheduler, scenario, n_episodes, label=None, seed_tag=None):
    """Run n_episodes of one scheduler on one scenario; return a list of rows."""
    env = FiveGSlicingEnv(scenario=scenario, weights=WEIGHTS)
    rows = []
    for ep in range(n_episodes):
        result = run_episode(scheduler, env, EVAL_SEED + ep)
        result.update({
            "scheduler": label or scheduler.name,
            "scenario": scenario,
            "episode": ep,
            "ppo_seed": seed_tag,
        })
        rows.append(result)
    return rows

# SUMMARISE
def summarise(episodes_df):
    """
    Turn the per-episode table into one row per scheduler/scenario with
    mean, standard deviation and 95% CI for every metric.
    """
    rows = []
    for (scenario, scheduler), group in episodes_df.groupby(
            ["scenario", "scheduler"], sort=False):
        row = {"scenario": scenario, "scheduler": scheduler, "n_episodes": len(group)}
        for m in METRICS:
            row[f"mean_{m}"] = group[m].mean()
            row[f"std_{m}"] = group[m].std(ddof=1)
            row[f"ci95_{m}"] = confidence_interval_95(group[m])
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary):
    #Print a readable results table: mean +/- 95% CI for key metrics
    print("\n" + "=" * 110)
    print("  EVALUATION RESULTS  (mean +/- 95% CI)")
    print("=" * 110)
    header = (f"  {'Scheduler':<24}{'Reward':>16}{'eMBB sat':>10}{'Lat ms':>8}"
              f"{'Viol':>7}{'Loss':>7}{'Power W':>9}{'Mbit/J':>8}{'Jain':>7}")
    for scenario in SCENARIOS:
        print(f"\n  --- {scenario} ---")
        print(header)
        part = summary[summary["scenario"] == scenario]
        for _, r in part.iterrows():
            reward = f"{r['mean_reward']:.2f} +/- {r['ci95_reward']:.2f}"
            print(f"  {r['scheduler']:<24}{reward:>16}"
                  f"{r['mean_embb_satisfaction']:10.3f}{r['mean_urllc_latency']:8.2f}"
                  f"{r['mean_urllc_violation_rate']:7.3f}{r['mean_mmtc_packet_loss']:7.3f}"
                  f"{r['mean_power_watts']:9.1f}{r['mean_energy_efficiency']:8.3f}"
                  f"{r['mean_jain_fairness']:7.3f}")


def significance_tests(episodes_df):
    """
    For each scenario, compare PPO's episode rewards using Welch's t-test
    against two references:
      (a) the best classical scheduler (RR, PF or Max-Throughput), and
      (b) Best Static, an "oracle" benchmark: it is chosen knowing the
          scenario in advance, unlike PPO 
    """
    classical = ["Round Robin", "Proportional Fair", "Max-Throughput"]
    rows = []
    for scenario in SCENARIOS:
        df = episodes_df[episodes_df["scenario"] == scenario]
        ppo = df[df["scheduler"] == "PPO"]["reward"]

        classical_df = df[df["scheduler"].isin(classical)]
        best_classical = classical_df.groupby("scheduler")["reward"].mean().idxmax()

        for comparison, name in [("best classical", best_classical),
                                 ("oracle", "Best Static")]:
            ref = df[df["scheduler"] == name]["reward"]
            t, p = welch_t_test(ppo, ref)
            rows.append({
                "scenario": scenario,
                "comparison": comparison,
                "reference": name,
                "ppo_mean_reward": ppo.mean(),
                "reference_mean_reward": ref.mean(),
                "difference": ppo.mean() - ref.mean(),
                "t_statistic": t,
                "p_value": p,
                "significant_at_0.05": p < 0.05,
            })
    return pd.DataFrame(rows)

# MAIN
def main(n_episodes):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []

    # Load all trained PPO seeds that exist 
    ppo_models = []
    for seed in PPO_SEEDS:
        path = os.path.join(MODEL_DIR, f"seed_{seed}", "best_model.zip")
        if os.path.exists(path):
            ppo_models.append((seed, PPOScheduler(path, "PPO")))
        else:
            print(f"  WARNING: no model found for seed {seed} ({path}), skipping")
    if not ppo_models:
        raise FileNotFoundError("No trained PPO models found. Run agents.train_ppo first.")
    print(f"\n  Loaded PPO seeds: {[s for s, _ in ppo_models]}")

    for scenario in SCENARIOS:
        print(f"\n  Evaluating scenario: {scenario}")

        # Best Static is chosen per scenario on separate traffic (seeds 1000+)
        best_action = find_best_static(scenario, weights=WEIGHTS)
        print(f"    Best Static action = {best_action} {ACTION_MAP[best_action]}")

        baselines = [
            RoundRobinScheduler(),
            ProportionalFairScheduler(),
            MaxThroughputScheduler(),
            StaticScheduler(best_action, "Best Static"),
        ]

        # PPO: every seed is run on the same 100 traffic episodes
        for seed, ppo in ppo_models:
            print(f"    PPO seed {seed} ...")
            all_rows += evaluate_scheduler(ppo, scenario, n_episodes,
                                           label="PPO", seed_tag=seed)

        for sched in baselines:
            print(f"    {sched.name} ...")
            all_rows += evaluate_scheduler(sched, scenario, n_episodes)

    # Build tables 
    episodes_df = pd.DataFrame(all_rows)
    summary = summarise(episodes_df)
    sig = significance_tests(episodes_df)

    # PPO results for each seed separately (shows run-to-run variation)
    ppo_only = episodes_df[episodes_df["scheduler"] == "PPO"]
    seed_summary = (ppo_only.groupby(["scenario", "ppo_seed"], sort=False)[METRICS]
                    .mean().reset_index())

    # Save 
    episodes_df.to_csv(os.path.join(RESULTS_DIR, "evaluation_episodes.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "evaluation_results.csv"), index=False)
    seed_summary.to_csv(os.path.join(RESULTS_DIR, "ppo_seed_summary.csv"), index=False)
    sig.to_csv(os.path.join(RESULTS_DIR, "significance_tests.csv"), index=False)

    # Print 
    print_summary(summary)

    print("\n  PPO per seed (mean reward / power W):")
    for _, r in seed_summary.iterrows():
        print(f"    {r['scenario']:<16} seed {int(r['ppo_seed'])}: "
              f"{r['reward']:7.2f} / {r['power_watts']:6.1f} W")

    print("\n  Significance: PPO vs references (Welch t-test)")
    for _, r in sig.iterrows():
        verdict = "significant" if r["significant_at_0.05"] else "not significant"
        print(f"    {r['scenario']:<16} vs {r['reference']:<18} "
              f"diff = {r['difference']:+6.2f}   p = {r['p_value']:.4f}  ({verdict})")

    print(f"\n  Saved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO vs baselines")
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    args = parser.parse_args()
    main(args.episodes)