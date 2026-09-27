"""
REINFORCEMENT LEARNING-BASED QUALITY OF SERVICE-AWARE 
RESOURCE SCHEDULING IN 5G NETWORK SLICING


A Custom OpenAI Gymnasium environment simulating a 5G network
It has three slice types: eMBB, URLLC, and mMTC.

"""

import gymnasium as gym        # Framework for Reinforcement Learning environment
import numpy as np              # For numerical operations
from gymnasium import spaces        # Defines observation and action spaces


class FiveGSlicingEnv(gym.Env):
    """
    Custom 5G Network Slicing Environment.
    
    Observation space: 5 normalised floats in [0, 1]
        [eMBB_throughput, URLLC_latency, mMTC_queue, energy, timestep]
    
    Action space: Discrete(10 PRB allocation strategies)
    
    Multi-objectiveReward function:
       r = w₁ * (throughput_eMBB) - w₂ * (packet_loss_mMTC) - w₃ * (latency_URLLC) - w₄ * (energy)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Environment parameters 
        self.total_prbs     = 100       # Total Physical Resource Blocks
        self.max_steps      = 200       # Steps per episode
        self.current_step   = 0

        #  Reward function weights (w1, w2, w3, w4) 
       
        self.w1 = 0.4   # eMBB throughput  (highest — eMBB is the primary objective. It has the PRB demand)
        self.w2 = 0.3   # URLLC latency    
        self.w3 = 0.2   # mMTC packet loss 
        self.w4 = 0.1   # Energy           (lowest  —  energy efficiency is the secondary objective)

        # QoS targets per slice 
        self.embb_max_throughput  = 100.0   # Mbps
        self.urllc_max_latency    = 10.0    # ms  (target: keep below this)
        self.mmtc_max_queue       = 50.0    # packets
        self.max_energy           = 100.0   # normalised energy units


        # Gymnasium spaces  
        # Action space - PRB allocation strategies
        # Each action = [eMBB_prbs, URLLC_prbs, mMTC_prbs]
        self.action_map = {
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

       
        # State space: 5 normalised floats all in [0, 1]
        # CUrrent state onservation by the PPO agent: [eMBB_throughput, URLLC_latency, mMTC_queue, energy, timestep]
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (5,),
            dtype = np.float32
        )

        # Action: choose one of 10 allocation strategies
        self.action_space = spaces.Discrete(10)

        # Internal state variables 
        self.embb_throughput  = 0.0
        self.urllc_latency    = 0.0
        self.mmtc_queue       = 0.0
        self.energy           = 0.0


    def reset(self, seed=None, options=None):
        #Reset environment to initial state at start of each episode
        super().reset(seed=seed)

        self.current_step     = 0
        self.embb_throughput  = 0.0
        self.urllc_latency    = self.urllc_max_latency * 0.5  # start at 50% latency
        self.mmtc_queue       = self.mmtc_max_queue   * 0.3   # start at 30% queue
        self.energy           = self.max_energy       * 0.2   # start at 20% energy

        observation = self._get_observation()
        info        = {}

        return observation, info


    def step(self, action):
        """
        Apply PRB allocation action, update slice states, compute reward.
        
        Args:
            action (int): Index into action_map (0-9)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # 1. Decode action into PRB allocations
        prb_alloc = self.action_map[int(action)]
        embb_prbs  = prb_alloc[0]
        urllc_prbs = prb_alloc[1]
        mmtc_prbs  = prb_alloc[2]

        # 2. Generate stochastic traffic (3GPP TR 36.814 models) 
        # eMBB: bursty Poisson arrivals (high volume, variable)
        embb_demand = np.random.poisson(lam=50)

        # URLLC: periodic low-volume with strict timing
        urllc_demand = np.random.poisson(lam=10)

        # mMTC: high-volume IoT sensor reporting
        mmtc_demand = np.random.poisson(lam=80)

        # 3. Update slice metrics based on PRB allocation 
        # eMBB throughput: proportional to PRBs allocated vs demand
        self.embb_throughput = min(
            (embb_prbs / max(embb_demand, 1)) * self.embb_max_throughput,
            self.embb_max_throughput
        )

        # URLLC latency: inverse relationship, i.e. more PRBs = lower latency
        self.urllc_latency = max(
            self.urllc_max_latency * (1 - urllc_prbs / self.total_prbs),
            0.1
        )

        # mMTC queue: packets accumulate if demand exceeds capacity
        served_mmtc = min(mmtc_prbs * 0.5, mmtc_demand)
        self.mmtc_queue = max(
            self.mmtc_queue + mmtc_demand - served_mmtc,
            0
        )
        self.mmtc_queue   = min(self.mmtc_queue, self.mmtc_max_queue)

        # Energy: proportional to total PRBs used
        self.energy = (sum(prb_alloc) / self.total_prbs) * self.max_energy

        # 4. Compute packet loss rate for mMTC 
        packet_loss = (self.mmtc_queue / self.mmtc_max_queue)

        # 5. Normalise all metrics to [0, 1] 
        norm_throughput   = self.embb_throughput / self.embb_max_throughput
        norm_latency      = self.urllc_latency   / self.urllc_max_latency
        norm_packet_loss  = packet_loss
        norm_energy       = self.energy           / self.max_energy

        # 6. Calculate multi-objective reward 
        reward = (
              self.w1 * norm_throughput
            - self.w2 * norm_latency
            - self.w3 * norm_packet_loss
            - self.w4 * norm_energy
        )

        # 7. Advance time step 
        self.current_step += 1
        terminated = False  # no early termination
        truncated  = self.current_step >= self.max_steps

        # 8. Build observation and info 
        observation = self._get_observation()
        info = {
            "embb_throughput" : self.embb_throughput,
            "urllc_latency"   : self.urllc_latency,
            "mmtc_queue"      : self.mmtc_queue,
            "energy"          : self.energy,
            "packet_loss"     : packet_loss,
            "reward"          : reward,
        }

        return observation, reward, terminated, truncated, info


    def _get_observation(self):
        #Build normalised observation vector
        return np.array([
            self.embb_throughput / self.embb_max_throughput,
            self.urllc_latency   / self.urllc_max_latency,
            self.mmtc_queue      / self.mmtc_max_queue,
            self.energy          / self.max_energy,
            self.current_step    / self.max_steps,
        ], dtype=np.float32)


    def render(self):
        """Print current environment state."""
        print(f"\nStep {self.current_step}/{self.max_steps}")
        print(f"  eMBB Throughput : {self.embb_throughput:.2f} Mbps")
        print(f"  URLLC Latency   : {self.urllc_latency:.2f} ms")
        print(f"  mMTC Queue      : {self.mmtc_queue:.2f} packets")
        print(f"  Energy          : {self.energy:.2f} units")


    def close(self):
        pass