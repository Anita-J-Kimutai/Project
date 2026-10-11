"""
REINFORCEMENT LEARNING-BASED QUALITY OF SERVICE-AWARE
RESOURCE SCHEDULING IN 5G NETWORK SLICING

Custom OpenAI Gymnasium environment simulating inter-slice PRB
allocation at a single 5G base station with three slices:
eMBB, URLLC and mMTC.

HOW THE SIMULATION WORKS 
    1. The scheduler (PPO agent or a baseline) picks an action.
       The action says how many PRBs each slice gets and how many
       PRBs are switched on in total (60, 70, 80, 90 or 100).
    2. New traffic arrives for each slice (random, Poisson).
    3. Each slice is served using the PRBs it was given:
        eMBB : demand that does not fit is simply lost
        URLLC: packets wait in a queue - creates latency
        mMTC : packets wait in a buffer - overflow = packet loss
    4. Base station power is calculated from how many PRBs are on.
    5. A reward is calculated that balances QoS against energy.
    6. The agent receives the new observation and the reward.

Time step = one slot of 0.5 ms (5G NR numerology 1, 30 kHz SCS).
"""

import queue

import gymnasium as gym          # RL environment framework
import numpy as np               # for numerical operations and random numbers
from gymnasium import spaces     # defines the shape of actions/observations


# TRAFFIC SCENARIOS
"""
Each scenario sets the AVERAGE amount of traffic per slot for each slice.
"lambda" is the mean of a Poisson distribution: every slot,
the actual traffic is a random number scattered around this mean
(e.g. lambda = 35 gives values like 29, 38, 33, 41 ...).

Units:
eMBB  -> number of PRBs the eMBB users would like this slot
URLLC -> number of small, urgent packets arriving this slot
mMTC  -> number of IoT sensor packets arriving this slot

"""
SCENARIOS = {
    "Balanced": {            # moderate load on all three slices
        "embb_lambda": 35,
        "urllc_lambda": 12,
        "mmtc_lambda": 40,
    },
    "eMBB-Dominant": {       # heavy video streaming demand
        "embb_lambda": 70,
        "urllc_lambda": 6,
        "mmtc_lambda": 24,
    },
    "URLLC-Critical": {      # spike in latency-sensitive traffic
        "embb_lambda": 25,
        "urllc_lambda": 40,
        "mmtc_lambda": 30,
    },
}

# ACTION SPACE
"""
An action has 2 parts:
   (a) a split pattern -> what share of the active PRBs each slice gets
   (b) a PRB level     -> how many PRBs are switched on in total
Combining 10 patterns with 5 levels gives 10 x 5 = 50 possible actions.
Allowing fewer than 100 PRBs is what lets the scheduler save energy:
PRBs that are not switched on do not draw transmit power.

Each pattern = [eMBB share, URLLC share, mMTC share]; shares add up to 1.0

"""
SPLIT_PATTERNS = [
    [0.60, 0.20, 0.20],   # eMBB-dominant
    [0.20, 0.60, 0.20],   # URLLC-dominant
    [0.20, 0.20, 0.60],   # mMTC-dominant
    [0.50, 0.30, 0.20],   # eMBB-URLLC balanced
    [0.50, 0.20, 0.30],   # eMBB-mMTC balanced
    [0.20, 0.50, 0.30],   # URLLC-mMTC balanced
    [0.40, 0.40, 0.20],   # eMBB-URLLC equal
    [0.40, 0.20, 0.40],   # eMBB-mMTC equal
    [0.20, 0.40, 0.40],   # URLLC-mMTC equal
    [0.34, 0.33, 0.33],   # Equal split
]

# Total number of PRBs switched on (out of 100 available)
PRB_LEVELS = [60, 70, 80, 90, 100]


def build_action_map():
    """
    Build the full action map: every split pattern at every PRB level.
    Returns a dictionary:  action index - [eMBB_prbs, URLLC_prbs, mMTC_prbs]
    Numbering rule:  action index = level_index * 10 + pattern_index
        actions  0-9  -> 60 PRBs  
        actions 10-19 -> 70 PRBs
        actions 20-29 -> 80 PRBs
        actions 30-39 -> 90 PRBs
        actions 40-49 -> 100 PRBs
    """
    action_map = {}

    # enumerate() gives both the position (index) and the value
    for level_idx, level in enumerate(PRB_LEVELS):
        for pat_idx, shares in enumerate(SPLIT_PATTERNS):
            # Convert shares into whole PRBs (PRBs cannot be fractional)
            urllc = int(round(shares[1] * level))
            mmtc = int(round(shares[2] * level))
            # Give eMBB whatever is left. Doing it this way (rather than
            # rounding eMBB too) guarantees the three numbers add up to
            # EXACTLY the PRB level, so no PRB is lost to rounding.
            embb = level - urllc - mmtc

            action_index_value = level_idx * len(SPLIT_PATTERNS) + pat_idx
            action_map[action_index_value] = [embb, urllc, mmtc]

    return action_map


# Built once when the file is imported, then shared by the environment
# and the baseline schedulers so they always use the same actions.
ACTION_MAP = build_action_map()


def action_index(pattern_idx, total_prbs=100):
    """
    Helper: find the action number for a given split pattern and PRB level.
    Used by the baseline schedulers and tests.
    Example: action_index(9, 80) -> 29  (equal split using 80 PRBs)
    """
    return PRB_LEVELS.index(total_prbs) * len(SPLIT_PATTERNS) + pattern_idx

# ENVIRONMENT CLASS

class FiveGSlicingEnv(gym.Env):
    """
    Custom 5G Network Slicing Environment 

    Observation (7 floats, each scaled to [0, 1]):
        [eMBB_demand, URLLC_arrivals, mMTC_arrivals,
         URLLC_queue, mMTC_queue, PRB_load, timestep]
        (demand/arrival values are from the previous slot, because a
         real scheduler decides before it knows the next slot's traffic)

    Action: Discrete(50) = 10 split patterns x 5 PRB levels

    Reward:
        r = w1 * eMBB_satisfaction
          - w2 * (URLLC_latency_norm + URLLC_SLA_violation)
          - w3 * mMTC_packet_loss
          - w4 * energy_norm
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, scenario=None, weights=(0.4, 0.3, 0.2, 0.1)):
        """
        Args:
            scenario: name from SCENARIOS, or None to pick a random
                      scenario at every reset. None is used for TRAINING,
                      so the agent experiences all three traffic
                      conditions and learns to adapt rather than
                      memorising one. 
            weights:  (w1, w2, w3, w4) reward weights. Passing them in
                      (rather than hard-coding) makes it easy to run the
                      weight sensitivity experiment later.
        """
        super().__init__()   # run gym.Env's own setup first

        # Stop early with a clear message if the scenario name is misspelt
        if scenario is not None and scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.fixed_scenario = scenario
        self.scenario = scenario or "Balanced"   # placeholder until reset()

        # Simulation parameters 
        self.total_prbs = 100
        self.max_steps = 200               # slots per episode (= 100 ms)
        self.slot_ms = 0.5                 # numerology 1 slot duration

        # How much traffic one PRB can carry in one slot 
        self.embb_mbps_per_prb = 1.0       # 100 PRBs - 100 Mbps peak 
        self.urllc_pkts_per_prb = 1.0      # URLLC packets are larger/robustly coded
        self.mmtc_pkts_per_prb = 2.0       # IoT packets are tiny, 2 fit per PRB

        # Queue limits and QoS targets 
        self.urllc_max_queue = 100.0       # packets
        self.mmtc_buffer = 100.0           # packets (anything beyond is dropped)
        self.urllc_latency_target = 1.0    # ms, URLLC SLA (3GPP target)
        self.urllc_max_latency = 10.0      # ms, used only to scale latency to 0-1

        # Base station power model 
        """
        A base station always draws some power just to stay on ,
        plus extra power that grows with how much it transmits.
        """
        self.p0_watts = 130.0              # static power at zero load
        self.delta_p = 4.7                 # how steeply power rises with load
        self.p_max_watts = 20.0            # maximum radio output power

        # Reward weights 
        self.w1, self.w2, self.w3, self.w4 = weights

        # Gymnasium spaces 
        """
        Tell Stable-Baselines3 what shape the inputs/outputs are,
        so it can build a neural network of the right size.
        """
        self.action_map = ACTION_MAP
        self.action_space = spaces.Discrete(len(self.action_map))   # 50 choices
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self._reset_state()

    """
    # Helper methods (the leading underscore marks them as
    # internal: they are used by this class, not called from outside)
    """

    def _reset_state(self):
        #Set all internal state variables to their starting values

        self.current_step = 0
        self.urllc_queue = 0.0            # packets waiting
        self.mmtc_queue = 0.0             # packets waiting
        self.last_embb_demand = 0.0       # traffic seen in the previous slot
        self.last_urllc_arrivals = 0.0
        self.last_mmtc_arrivals = 0.0
        self.last_load = 1.0              # assume all PRBs on at the start

    def power_watts(self, active_prbs):
        """
        Base station input power for a given number of active PRBs.

        Auer et al. (2011):  P_in = P0 + delta_p * P_out
        where P_out = P_max * (active PRBs / total PRBs).

        Worked example with 80 PRBs on:
            load  = 80 / 100            = 0.8
            P_out = 20 W * 0.8          = 16 W
            P_in  = 130 + 4.7 * 16      = 205.2 W
        With all 100 PRBs on, P_in = 224 W, so switching off 20 PRBs
        saves about 19 W (around 8%).
        """
        load = active_prbs / self.total_prbs
        return self.p0_watts + self.delta_p * self.p_max_watts * load

    def _get_observation(self):
        """
        Build the observation vector the agent sees.

        Every value is divided by a "typical maximum" so it falls
        roughly between 0 and 1. Neural networks learn much better when
        inputs are on the same scale; otherwise a value like 70 (PRBs)
        would dominate a value like 0.3 (load) because of its magnitude
        """
        obs = np.array([
            self.last_embb_demand / 100.0,      # eMBB demand (max 100 PRBs)
            self.last_urllc_arrivals / 60.0,    # URLLC arrivals
            self.last_mmtc_arrivals / 150.0,    # mMTC arrivals
            self.urllc_queue / self.urllc_max_queue,
            self.mmtc_queue / self.mmtc_buffer,
            self.last_load,                      # already 0-1
            self.current_step / self.max_steps,  # progress through episode
        ], dtype=np.float32)
        """
        Random traffic can occasionally exceed the "typical maximum"
        (e.g. a rare burst of 160 mMTC packets). np.clip caps every
        value at 0-1 so the observation never breaks the declared space.
        return np.clip(obs, 0.0, 1.0)
        """

    # Gymnasium interface (the methods SB3 calls)
   

    def reset(self, seed=None, options=None):
        """
        Reset to the start of an episode.

        options={"scenario": name} overrides the scenario for this episode
        (used during evaluation to force a particular scenario).
        """
        # Calling the parent reset with a seed initialises self.np_random,
        # the environment's own random number generator. All traffic is
        # drawn from it, so the same seed always produces the same traffic.
        super().reset(seed=seed)

        # Decide which scenario this episode uses, in order of priority:
        if options and "scenario" in options:
            self.scenario = options["scenario"]          # explicit override
        elif self.fixed_scenario is not None:
            self.scenario = self.fixed_scenario          # set at creation
        else:
            names = list(SCENARIOS.keys())               # random (training)
            self.scenario = names[self.np_random.integers(len(names))]

        self._reset_state()

        # Gymnasium requires reset() to return (observation, info)
        return self._get_observation(), {"scenario": self.scenario}

    def step(self, action):
        """
        Simulate one 0.5 ms slot.

        Returns the five values Gymnasium requires:
            observation, reward, terminated, truncated, info
        """

        # 1. Decode the action into PRBs per slice 
        """
        int(action) is needed because SB3 passes a NumPy number,
        which does not work directly as a dictionary key
        """
        embb_prbs, urllc_prbs, mmtc_prbs = self.action_map[int(action)]
        active_prbs = embb_prbs + urllc_prbs + mmtc_prbs

        # 2. Generate this slot's random traffic
        cfg = SCENARIOS[self.scenario]
        embb_demand = self.np_random.poisson(cfg["embb_lambda"])
        urllc_arrivals = self.np_random.poisson(cfg["urllc_lambda"])
        mmtc_arrivals = self.np_random.poisson(cfg["mmtc_lambda"])

        # 3. eMBB: serve as much demand as the PRBs allow
        """ 
        eMBB has no queue: video data that cannot be sent this slot is
        treated as lost throughput (the user just sees lower quality).
        min() means: you cannot serve more than you were given,
        and you cannot serve more than was asked for
        """
        embb_served_prbs = min(embb_prbs, embb_demand)
        embb_throughput = embb_served_prbs * self.embb_mbps_per_prb     # Mbps

        # Satisfaction = fraction of demand that was served (0 to 1).
        # If no demand arrived, nothing was missed, so satisfaction = 1.
        embb_satisfaction = embb_served_prbs / embb_demand if embb_demand > 0 else 1.0

        # 4. URLLC: queue -> serve -> estimate delay
        # (a) new packets join the back of the queue
        self.urllc_queue += urllc_arrivals

        # (b) if the queue overflows, the excess is dropped
        urllc_dropped = max(self.urllc_queue - self.urllc_max_queue, 0.0)
        self.urllc_queue = min(self.urllc_queue, self.urllc_max_queue)

        # (c) serve as many packets as the URLLC PRBs can carry
        urllc_capacity = urllc_prbs * self.urllc_pkts_per_prb
        urllc_served = min(self.urllc_queue, urllc_capacity)
        self.urllc_queue -= urllc_served

        # (d) estimate latency. 
        """
        # Every packet needs at least one slot(0.5 ms).
        # Packets still waiting add extra slots: if 20 are
        # left and 10 can be served per slot, it takes 2 more slots
        # to clear them. So:
        # latency = slot * (1 + remaining_queue / capacity)
        # Example: 0.5 * (1 + 20/10) = 1.5 ms -> violates the 1 ms SLA.
        # max(..., 1.0) avoids dividing by zero if URLLC got no PRBs.
        """
        urllc_latency = self.slot_ms * (1 + self.urllc_queue / max(urllc_capacity, 1.0))
        urllc_latency = min(urllc_latency, self.urllc_max_latency)   # cap at 10 ms

        # (e) did this slot break the URLLC SLA? 1 = yes, 0 = no
        urllc_violation = 1.0 if urllc_latency > self.urllc_latency_target else 0.0

        # 5. mMTC: finite buffer, overflow = packet loss
        """
        Same queue idea as URLLC, but IoT traffic is delay-tolerant,
        so what matters is not latency but packets lost to overflow.
        """
        self.mmtc_queue += mmtc_arrivals
        mmtc_dropped = max(self.mmtc_queue - self.mmtc_buffer, 0.0)
        self.mmtc_queue = min(self.mmtc_queue, self.mmtc_buffer)
        mmtc_served = min(self.mmtc_queue, mmtc_prbs * self.mmtc_pkts_per_prb)
        self.mmtc_queue -= mmtc_served

        # Loss rate = share of this slot's arrivals that were dropped
        mmtc_loss = mmtc_dropped / mmtc_arrivals if mmtc_arrivals > 0 else 0.0

        # 6. Energy (Auer et al., 2011) 
        power = self.power_watts(active_prbs)                 # Watts
        # Energy = power x time. slot_ms / 1000 converts ms to seconds.
        energy_joules = power * self.slot_ms / 1000.0         # Joules per slot
        """
        For the reward we only use the part the scheduler can control:
        the load-dependent share (0.6 to 1.0). The static 130 W is
        drawn regardless of the scheduling decision.
        """
        energy_norm = active_prbs / self.total_prbs

        # 7. Compute the reward
        latency_norm = urllc_latency / self.urllc_max_latency   # 0.05 to 1.0

        # URLLC penalty = normalised latency + SLA violation 
        """
        Latency alone is a weak signal (1.5 ms only scores 0.15), so
        the agent would barely notice breaking the SLA. Adding the
        violation flag makes missing the 1 ms target cost far more
        (SLA-based penalty, following Sherif et al., 2024).
        """
        urllc_penalty = latency_norm + urllc_violation

        # Positive term rewards good eMBB service; negative terms
        # penalise URLLC delay, mMTC loss and energy use.
        reward = (
              self.w1 * embb_satisfaction
            - self.w2 * urllc_penalty
            - self.w3 * mmtc_loss
            - self.w4 * energy_norm
        )

        # 8. Remember this slot's traffic for the next observation
        self.last_embb_demand = embb_demand
        self.last_urllc_arrivals = urllc_arrivals
        self.last_mmtc_arrivals = mmtc_arrivals
        self.last_load = energy_norm

        # 9. Advance time and check whether the episode is over
        self.current_step += 1
        terminated = False   # no "failure" state ends an episode early
        # truncated = True when the time limit (200 slots) is reached
        truncated = self.current_step >= self.max_steps

        # 10. Extra metrics for evaluation
        """
        The agent never sees "info"; it is used by evaluate.py and the
        dashboard to report per-slice results
        """
        info = {
            "scenario": self.scenario,
            "embb_prbs": embb_prbs,
            "urllc_prbs": urllc_prbs,
            "mmtc_prbs": mmtc_prbs,
            "active_prbs": active_prbs,
            "embb_throughput": embb_throughput,
            "embb_satisfaction": embb_satisfaction,
            "urllc_latency": urllc_latency,
            "urllc_violation": urllc_violation,
            "urllc_dropped": urllc_dropped,
            "mmtc_queue": self.mmtc_queue,
            "packet_loss": mmtc_loss,
            "power_watts": power,
            "energy_joules": energy_joules,
            "energy": energy_norm,
            "reward": reward,
        }
        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        """Print the current state (useful for debugging)."""
        print(f"\nStep {self.current_step}/{self.max_steps} [{self.scenario}]")
        print(f"  URLLC queue : {self.urllc_queue:.1f} packets")
        print(f"  mMTC queue  : {self.mmtc_queue:.1f} packets")
        print(f"  PRB load    : {self.last_load:.2f}")

    def close(self):
        pass