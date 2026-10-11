"""
REINFORCEMENT LEARNING-BASED QUALITY OF SERVICE-AWARE
RESOURCE SCHEDULING IN 5G NETWORK SLICING

BASELINE SCHEDULERS 

Rule-based schedulers used as benchmarks for the PPO agent.

    1. Round Robin       - rotates slice priority; ignores traffic
    2. Proportional Fair - favours the slice whose demand is high
                           relative to what it has recently received
    3. Max-Throughput    - greedy: picks the split that serves the most
                           traffic right now; ignores latency and fairness
    4. Best Static       - the single fixed allocation with the
                           highest reward for a scenario, found by
                           testing all 50 actions. A strong benchmark for PPO 

    Classical schedulers (1-3) are not energy-aware, they always
    switch on all 100 PRBs. This matches how they work in real
    networks 

All schedulers share the same interface:
    allocate(observation) - action index (0-49)
"""

import numpy as np

from environment.env_5g import (
    FiveGSlicingEnv, ACTION_MAP, SCENARIOS, SPLIT_PATTERNS, action_index
)

# Pattern numbers from SPLIT_PATTERNS in env_5g.py, named for readability
EMBB_DOMINANT, URLLC_DOMINANT, MMTC_DOMINANT = 0, 1, 2
EQUAL_SPLIT = 9


def demand_from_observation(obs):
    """
    Convert the agent's normalised observation back into an estimate of
    how many PRBs each slice needs to be used by rule-based schedulers.

    Observation layout (as in env_5g.py):
        [0] eMBB demand /100   [1] URLLC arrivals /60   [2] mMTC arrivals /150
        [3] URLLC queue /100   [4] mMTC queue /100      [5] load  [6] time

    URLLC needs 1 PRB per packet; mMTC fits 2 packets per PRB, so its
    packet count is halved. Queued packets also need serving, so they
    are added to new arrivals.
    """
    embb = obs[0] * 100
    urllc = obs[1] * 60 + obs[3] * 100
    mmtc = (obs[2] * 150 + obs[4] * 100) / 2
    return np.array([embb, urllc, mmtc])


# BASE CLASS
class BaseScheduler:
    #Parent class: every scheduler must implement allocate()

    def __init__(self, name):
        self.name = name

    def allocate(self, obs):
        raise NotImplementedError(f"{self.name} must implement allocate()")

    def reset(self):
        """Clear any memory at the start of an episode (default: none)."""
        pass

    def __str__(self):
        return f"Scheduler({self.name})"

# 1. ROUND ROBIN
class RoundRobinScheduler(BaseScheduler):
    """
    Gives each slice the dominant share in turn (eMBB -> URLLC -> mMTC
    -> repeat), regardless of traffic. Fair over time, but blind to
    demand, so it often gives PRBs to a slice that does not need them
    """

    def __init__(self):
        super().__init__("Round Robin")
        # The three slice-dominant patterns, all using 100 PRBs
        self.sequence = [action_index(p, 100)
                         for p in (EMBB_DOMINANT, URLLC_DOMINANT, MMTC_DOMINANT)]
        self.index = 0

    def allocate(self, obs):
        # "%" (modulo) wraps the index back to 0 after the last item
        action = self.sequence[self.index % len(self.sequence)]
        self.index += 1
        return action

    def reset(self):
        self.index = 0

# 2. PROPORTIONAL FAIR
class ProportionalFairScheduler(BaseScheduler):
    """
    Classic PF rule:  priority_i = current demand_i / average received_i

    A slice that has been under-served recently has a small average, so
    its priority rises and it gets the dominant share next. This
    balances efficiency (serve high demand) with fairness
    The average is an exponential moving average (EMA).
    """

    def __init__(self, alpha=0.1):
        #alpha: EMA smoothing; higher = reacts faster to recent slots
        super().__init__("Proportional Fair")
        self.alpha = alpha
        self.reset()

    def allocate(self, obs):
        demand = demand_from_observation(obs)

        # +1e-6 avoids dividing by zero
        priority = demand / (self.avg_received + 1e-6)
        dominant = int(np.argmax(priority))      # 0=eMBB, 1=URLLC, 2=mMTC
        action = action_index(dominant, 100)     # pattern 0/1/2 at 100 PRBs

        # Update the EMA with the PRBs each slice actually received
        received = np.array(ACTION_MAP[action], dtype=float)
        self.avg_received = (1 - self.alpha) * self.avg_received + self.alpha * received
        return action

    def reset(self):
        # Start as if each slice had an equal share (100/3 PRBs)
        self.avg_received = np.full(3, 100 / 3)


# 3. MAX-THROUGHPUT
class MaxThroughputScheduler(BaseScheduler):
    """
    Greedy: tries all 10 split patterns (at 100 PRBs) and picks the one
    that would serve the most traffic given last slot's demand.
    It maximises the total amount served, with no regard for URLLC
    latency targets or fairness between slices.
    """
    def __init__(self):
        super().__init__("Max-Throughput")
        self.candidates = [action_index(p, 100) for p in range(len(SPLIT_PATTERNS))]

    def allocate(self, obs):
        demand = demand_from_observation(obs)
        best_action, best_served = self.candidates[0], -1.0
        for a in self.candidates:
            alloc = np.array(ACTION_MAP[a])
            # A slice cannot use more PRBs than it needs: min() per slice
            served = np.minimum(alloc, demand).sum()
            if served > best_served:
                best_action, best_served = a, served
        return best_action



# 4. BEST STATIC (benchmark)


class StaticScheduler(BaseScheduler):
    #Always plays the same action. Used for the Best Static benchmark

    def __init__(self, action, name=None):
        super().__init__(name or f"Static (action {action})")
        self.action = action

    def allocate(self, obs):
        return self.action


def find_best_static(scenario, weights=(0.4, 0.3, 0.2, 0.1),
                     episodes=5, seed=1000):
    """
    Test all 50 actions as fixed allocations on one scenario and return
    the action with the highest mean episode reward.

    Uses seeds 1000+ so the search never sees the evaluation traffic
    (evaluation uses seed 42+); choosing on the test data would be unfair
    to the other schedulers.
    """
    env = FiveGSlicingEnv(scenario=scenario, weights=weights)
    best_action, best_reward = None, -np.inf
    for a in ACTION_MAP:
        total = 0.0
        for ep in range(episodes):
            env.reset(seed=seed + ep)
            done = False
            while not done:
                _, r, term, trunc, _ = env.step(a)
                total += r
                done = term or trunc
        mean_reward = total / episodes
        if mean_reward > best_reward:
            best_action, best_reward = a, mean_reward
    return best_action



# RUN A SCHEDULER (shared by the quick test and evaluate.py)
def run_scheduler(scheduler, scenario, n_episodes=100, seed=42,
                  weights=(0.4, 0.3, 0.2, 0.1)):
    """
    Run one scheduler on one scenario for n_episodes and return the
    mean and standard deviation of each metric (per-episode averages)
    """
    env = FiveGSlicingEnv(scenario=scenario, weights=weights)
    metrics = ["embb_throughput", "embb_satisfaction", "urllc_latency",
               "urllc_violation", "packet_loss", "power_watts", "active_prbs"]
    per_episode = {m: [] for m in metrics}
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        scheduler.reset()
        slot_values = {m: [] for m in metrics}
        ep_reward, done = 0.0, False
        while not done:
            action = scheduler.allocate(obs)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            for m in metrics:
                slot_values[m].append(info[m])
            done = term or trunc
        rewards.append(ep_reward)
        for m in metrics:
            per_episode[m].append(np.mean(slot_values[m]))

    result = {"scheduler": scheduler.name, "scenario": scenario,
              "mean_reward": np.mean(rewards), "std_reward": np.std(rewards)}
    for m in metrics:
        result[f"mean_{m}"] = np.mean(per_episode[m])
        result[f"std_{m}"] = np.std(per_episode[m])
    return result

# QUICK TEST
if __name__ == "__main__":
    print("\n" + "=" * 86)
    print("  BASELINE SCHEDULER QUICK TEST (10 episodes per scenario)")
    print("=" * 86)
    print(f"  {'Scenario':<16}{'Scheduler':<26}{'Reward':>8}{'eMBB sat':>9}"
          f"{'Viol':>7}{'Loss':>7}{'Power W':>9}")

    for scenario in SCENARIOS:
        best = find_best_static(scenario)
        schedulers = [
            RoundRobinScheduler(),
            ProportionalFairScheduler(),
            MaxThroughputScheduler(),
            StaticScheduler(best, f"Best Static {ACTION_MAP[best]}"),
        ]
        for s in schedulers:
            r = run_scheduler(s, scenario, n_episodes=10)
            print(f"  {scenario:<16}{s.name:<26}{r['mean_reward']:8.2f}"
                  f"{r['mean_embb_satisfaction']:9.3f}{r['mean_urllc_violation']:7.3f}"
                  f"{r['mean_packet_loss']:7.3f}{r['mean_power_watts']:9.1f}")
        print()

    print("All schedulers tested successfully.")