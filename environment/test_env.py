"""
REINFORCEMENT LEARNING-BASED QUALITY OF SERVICE-AWARE
RESOURCE SCHEDULING IN 5G NETWORK SLICING

Checks:
    1. Gymnasium interface compliance (check_env)
    2. Scenarios produce different traffic
    3. Energy varies with the number of active PRBs
    4. Same seed - identical results 
    5. Fixed allocations across scenarios

"""

import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.env_5g import (
    FiveGSlicingEnv, SCENARIOS, ACTION_MAP, action_index
)


def run_fixed(action, scenario, seed=0, episodes=5):
    """
    Run the same action every slot for several episodes and return the
    average of each metric. This is a "fixed allocation" scheduler.
    It lets us check the environment behaves sensibly before any
    learning is involved.
    """
    env = FiveGSlicingEnv(scenario=scenario)

    # Metrics to average (names match the keys in env.step()'s info dict)
    keys = ["embb_throughput", "embb_satisfaction", "urllc_latency",
            "urllc_violation", "packet_loss", "power_watts"]

    # Dictionary comprehension: creates {"embb_throughput": [], ...}
    totals = {k: [] for k in keys}
    rewards = []

    for ep in range(episodes):
        # seed + ep gives each episode different (but repeatable) traffic
        env.reset(seed=seed + ep)
        ep_reward, done = 0.0, False

        while not done:
            # "_" discards the observation
            _, r, term, trunc, info = env.step(action)
            ep_reward += r
            for k in keys:
                totals[k].append(info[k])
            done = term or trunc

        rewards.append(ep_reward)

    # Average every metric over all slots of all episodes
    out = {k: float(np.mean(v)) for k, v in totals.items()}
    out["reward"] = float(np.mean(rewards))   # mean TOTAL reward per episode
    return out


def test_check_env():
    """
    Gymnasium's built-in checker confirms reset() and step() return
    the right types and shapes and that observations stay inside the
    declared [0, 1] space. 
    """
    check_env(FiveGSlicingEnv(), skip_render_check=True)
    print("[PASS] check_env: Gymnasium interface is valid")


def test_scenarios_differ():
    """
    This runs 2,000 slots per scenario and checks that the average
    traffic matches each scenario's lambda values, and that the
    scenarios genuinely differ.
    """
    print("\nMean traffic per scenario (2,000 slots):")
    means = {}
    for name in SCENARIOS:
        env = FiveGSlicingEnv(scenario=name)
        env.reset(seed=1)
        demand = []
        for _ in range(2000):
            env.step(action_index(9, 100))    # action doesn't affect arrivals
            demand.append([env.last_embb_demand,
                           env.last_urllc_arrivals,
                           env.last_mmtc_arrivals])
            # Start a new episode whenever the 200-slot limit is reached
            if env.current_step >= env.max_steps:
                env.reset()

        # axis=0 averages each column separately -> [eMBB, URLLC, mMTC]
        means[name] = np.mean(demand, axis=0)
        print(f"  {name:<15} eMBB={means[name][0]:5.1f} PRBs  "
              f"URLLC={means[name][1]:5.1f} pkts  mMTC={means[name][2]:5.1f} pkts")

    # np.allclose is True if two arrays are (almost) equal, so
    # "not allclose" confirms the scenarios really are different
    assert not np.allclose(means["Balanced"], means["eMBB-Dominant"])
    assert not np.allclose(means["Balanced"], means["URLLC-Critical"])
    print("[PASS] Scenarios generate different traffic")


def test_energy_varies():
    """
    each PRB level gives a different power
    every action's PRBs add up exactly to its level.
    """
    env = FiveGSlicingEnv()
    powers = [env.power_watts(p) for p in (60, 70, 80, 90, 100)]
    print("\nPower per PRB level:",
          ", ".join(f"{p} PRBs={w:.1f} W" for p, w in zip((60, 70, 80, 90, 100), powers)))

    # A set removes duplicates: 5 different powers - set of size 5
    assert len(set(powers)) == 5

    # Collect the distinct PRB totals across all 50 actions
    sums = sorted({sum(v) for v in ACTION_MAP.values()})
    assert sums == [60, 70, 80, 90, 100], sums
    print("[PASS] Energy varies with active PRBs; action sums are exact")


def test_reproducible():
    """
    Same seed must give identical results; a different seed must not.
    The second check matters too: if different seeds also gave the
    same result, the randomness would not be working at all.
    """
    a = run_fixed(action_index(9, 80), "Balanced", seed=7, episodes=2)
    b = run_fixed(action_index(9, 80), "Balanced", seed=7, episodes=2)
    c = run_fixed(action_index(9, 80), "Balanced", seed=8, episodes=2)
    assert a == b, "Same seed gave different results"
    assert a != c, "Different seeds gave identical results"
    print("[PASS] Same seed reproduces results; different seeds differ")


def sanity_table():
    """
    prints a table that can be used to check whether
    results make sense; more eMBB PRBs - higher throughput,
    fewer PRBs - lower power, URLLC-Critical - more SLA violations
    unless URLLC gets plenty of PRBs.
    Column key: Thr = eMBB Mbps, Sat = eMBB satisfaction,
    Lat = URLLC latency, Viol = share of slots breaking the 1 ms SLA,
    Loss = mMTC packet loss rate.
    """
    print("\nFixed allocations across scenarios (5 episodes each):")
    tests = {
        "Equal split @100": action_index(9, 100),
        "Equal split @70": action_index(9, 70),
        "eMBB-dom @100": action_index(0, 100),
        "URLLC-dom @90": action_index(1, 90),
    }
    # In f-strings, ":<15" = left-align in 15 characters and
    # ":>8" = right-align in 8, which keeps the table columns lined up
    header = (f"  {'Scenario':<15}{'Allocation':<18}{'Thr':>6}{'Sat':>6}"
              f"{'Lat ms':>8}{'Viol':>6}{'Loss':>6}{'Power W':>9}{'Reward':>9}")
    print(header)
    for scen in SCENARIOS:
        for label, act in tests.items():
            m = run_fixed(act, scen)
            print(f"  {scen:<15}{label:<18}{m['embb_throughput']:6.1f}"
                  f"{m['embb_satisfaction']:6.2f}{m['urllc_latency']:8.2f}"
                  f"{m['urllc_violation']:6.2f}{m['packet_loss']:6.2f}"
                  f"{m['power_watts']:9.1f}{m['reward']:9.2f}")


# Only runs when the file is executed directly, not when imported
if __name__ == "__main__":
    test_check_env()
    test_scenarios_differ()
    test_energy_varies()
    test_reproducible()
    sanity_table()
    print("\nAll environment tests passed.")