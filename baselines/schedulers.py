"""
REINFORCEMENT LEARNING-BASED QUALITY OF SERVICE-AWARE 
RESOURCE SCHEDULING IN 5G NETWORK SLICING

Description:
    Implements three traditional rule-based PRB scheduling
    algorithms used as baselines for comparison against
    the PPO reinforcement learning agent.

    Schedulers implemented:
        1. Round Robin       — equal rotation, no QoS awareness
        2. Proportional Fair — weighted by historical throughput
        3. Max-Throughput    — greedy, highest channel quality

    All schedulers follow the same interface:
        allocate(state) -> action (int, index into action_map)

"""

import numpy as np


# Exactly as in env_5g.py
# Each key is an action index (0-9)
# Each value is [eMBB_prbs, URLLC_prbs, mMTC_prbs]
ACTION_MAP = {
    0: [60, 20, 20],   # eMBB-dominant
    1: [20, 60, 20],   # URLLC-dominant
    2: [20, 20, 60],   # mMTC-dominant
    3: [50, 30, 20],   # eMBB-URLLC balanced
    4: [50, 20, 30],   # eMBB-mMTC balanced
    5: [20, 50, 30],   # URLLC-mMTC balanced
    6: [40, 40, 20],   # eMBB-URLLC equal
    7: [40, 20, 40],   # eMBB-mMTC equal
    8: [20, 40, 40],   # URLLC-mMTC equal
    9: [34, 33, 33],   # Equal split
}


# BASE CLASS 

class BaseScheduler:
    """
    Abstract base class for all baseline schedulers.
    All schedulers must implement the allocate() method.
    """

    def __init__(self, name):
        """
        Args:
            name (str): human-readable scheduler name
        """
        self.name = name

    def allocate(self, state):
        """
        Select a PRB allocation action given the current state.

        Args:
            state (np.array): current normalised observation
                              [throughput, latency, queue,
                               energy, timestep]

        Returns:
            action (int): index into ACTION_MAP (0-9)
        """
        raise NotImplementedError(
            f"{self.name} must implement allocate()"
        )

    def reset(self):
        """Reset any internal state at the start of an episode."""
        pass

    def __str__(self):
        return f"Scheduler({self.name})"


# ROUND ROBIN 

class RoundRobinScheduler(BaseScheduler):
    """
    Round Robin Scheduler.

    Rotates through a fixed sequence of balanced allocation
    strategies at each time step, ensuring equal treatment
    of all slices regardless of current demand or channel
    conditions. Provides fairness but no QoS differentiation.

    Reference: Döttling et al. (2009)
    """

    def __init__(self):
        super().__init__("Round Robin")

        # Sequence of actions that cycle through balanced
        # allocations — favouring equal treatment
        self.action_sequence = [9, 6, 7, 8, 9, 6, 7, 8, 9, 6]
        self.current_index   = 0

    def allocate(self, state):
        """
        Return the next action in the rotation sequence.
        State is ignored — Round Robin is demand-unaware.
        """
        action = self.action_sequence[
            self.current_index % len(self.action_sequence)
        ]
        self.current_index += 1
        return action

    def reset(self):
        #Reset rotation index at episode start
        self.current_index = 0


# PROPORTIONAL FAIR 

class ProportionalFairScheduler(BaseScheduler):
    """
    Proportional Fair Scheduler.

    Weights PRB allocation by the ratio of each slice's
    current channel quality (throughput) to its historical
    average throughput. Balances efficiency and fairness
    and is the most widely deployed scheduler in LTE and
    early 5G systems.

    """

    def __init__(self, alpha=0.1):
        """
        Args:
            alpha (float): smoothing factor for historical
                           average update (0 < alpha < 1)
        """
        super().__init__("Proportional Fair")
        self.alpha = alpha

        # Historical average throughput per slice
        # Initialised to 1.0 to avoid division by zero
        self.avg_throughput = np.array([1.0, 1.0, 1.0])

    def allocate(self, state):
        """
        Select action based on PF scoring:
        score_i = current_rate_i / avg_throughput_i

        Args:
            state: [throughput_eMBB, latency_URLLC,
                    queue_mMTC, energy, timestep]

        Returns:
            action (int): action index with allocation most
                          proportional to current demands
        """
        # Extract current rates from state
        # state[0] = eMBB throughput (normalised)
        # state[1] = URLLC latency  (inverse)
        # state[2] = mMTC queue     (inverse)
        current_rates = np.array([
            state[0],          # eMBB: higher throughput = higher priority
            1 - state[1],      # URLLC: lower latency = higher priority
            1 - state[2],      # mMTC: lower queue = higher priority
        ])

        # Compute PF scores — ratio of current to historical
        pf_scores = current_rates / (self.avg_throughput + 1e-8)

        # Update historical averages using exponential moving average
        self.avg_throughput = (
            (1 - self.alpha) * self.avg_throughput
            + self.alpha * current_rates
        )

        # Find the action whose allocation best matches PF scores
        # The slice with highest PF score gets most PRBs
        dominant_slice = int(np.argmax(pf_scores))

        """
        Map dominant slice to corresponding action
        0=eMBB dominant, 1=URLLC dominant, 2=mMTC dominant
        Actions 0,1,2 are slice-dominant
        """
        action = dominant_slice 

        return action

    def reset(self):
        """Reset historical averages at episode start."""
        self.avg_throughput = np.array([1.0, 1.0, 1.0])


# ══ MAX THROUGHPUT ══════════════════════════════════════════

class MaxThroughputScheduler(BaseScheduler):
    """
    Max-Throughput Scheduler.

    Greedily allocates all PRBs to the slice with the
    highest instantaneous channel quality (throughput),
    maximising aggregate network throughput at the expense
    of fairness and QoS differentiation across slices.

    """

    def __init__(self):
        super().__init__("Max-Throughput")

    def allocate(self, state):
        """
        Select the action that maximises throughput for
        the slice currently experiencing the best conditions.

        Args:
            state: [throughput_eMBB, latency_URLLC,
                    queue_mMTC, energy, timestep]

        Returns:
            action (int): dominant-slice action (0, 1, or 2)
        """
        # Score each slice by its current performance metric
        # Higher score = better current conditions
        slice_scores = np.array([
            state[0],          # eMBB: higher throughput = better
            1 - state[1],      # URLLC: lower latency = better
            1 - state[2],      # mMTC: lower queue = better
        ])

        # Greedily assign to highest-scoring slice
        best_slice = int(np.argmax(slice_scores))

        # Actions 0,1,2 = eMBB, URLLC, mMTC dominant respectively
        action = best_slice

        return action

    def reset(self):
        #No internal state to reset
        pass


# UTILITY — RUN A SCHEDULER FOR N EPISODES

def run_scheduler(scheduler, env, n_episodes=100, seed=42):
    """
    Run a baseline scheduler for N episodes and collect metrics.

    Args:
        scheduler  : BaseScheduler instance
        env        : FiveGSlicingEnv instance
        n_episodes : number of evaluation episodes
        seed       : random seed for reproducibility

    Returns:
        dict containing mean metrics across all episodes
    """
    all_rewards      = []
    all_throughputs  = []
    all_latencies    = []
    all_packet_losses= []
    all_energies     = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        scheduler.reset()

        ep_reward       = 0
        ep_throughput   = []
        ep_latency      = []
        ep_packet_loss  = []
        ep_energy       = []

        done = False

        while not done:
            # Scheduler selects action based on current state
            action = scheduler.allocate(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Collect metrics from info dict
            ep_reward      += reward
            ep_throughput  .append(info.get("embb_throughput", 0))
            ep_latency     .append(info.get("urllc_latency",   0))
            ep_packet_loss .append(info.get("packet_loss",     0))
            ep_energy      .append(info.get("energy",          0))

            done = terminated or truncated

        # Store episode averages
        all_rewards     .append(ep_reward)
        all_throughputs .append(np.mean(ep_throughput))
        all_latencies   .append(np.mean(ep_latency))
        all_packet_losses.append(np.mean(ep_packet_loss))
        all_energies    .append(np.mean(ep_energy))

    # Return summary statistics
    return {
        "scheduler"     : scheduler.name,
        "mean_reward"   : np.mean(all_rewards),
        "std_reward"    : np.std(all_rewards),
        "mean_throughput": np.mean(all_throughputs),
        "std_throughput" : np.std(all_throughputs),
        "mean_latency"  : np.mean(all_latencies),
        "std_latency"   : np.std(all_latencies),
        "mean_packet_loss": np.mean(all_packet_losses),
        "std_packet_loss" : np.std(all_packet_losses),
        "mean_energy"   : np.mean(all_energies),
        "std_energy"    : np.std(all_energies),
    }


# QUICK TEST

if __name__ == "__main__":
    """Quick test to verify all schedulers work correctly."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    from environment.env_5g import FiveGSlicingEnv

    env = FiveGSlicingEnv()
    schedulers = [
        RoundRobinScheduler(),
        ProportionalFairScheduler(),
        MaxThroughputScheduler(),
    ]

    print("\n" + "="*50)
    print("  BASELINE SCHEDULER QUICK TEST (5 episodes)")
    print("="*50)

    for scheduler in schedulers:
        results = run_scheduler(scheduler, env, n_episodes=5)
        print(f"\n{scheduler.name}:")
        print(f"  Mean reward    : {results['mean_reward']:.4f}")
        print(f"  Mean throughput: {results['mean_throughput']:.4f}")
        print(f"  Mean latency   : {results['mean_latency']:.4f}")
        print(f"  Mean energy    : {results['mean_energy']:.4f}")

    env.close()
    print("\n✅ All schedulers tested successfully!")