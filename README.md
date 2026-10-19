## REINFORCEMENT LEARNING-BASED QUALITY OF SERVICE-AWARE RESOURCE SCHEDULING IN 5G NETWORK SLICING

A single-agent Reinforcement Learning scheduler traine using Proximal Policy Optimization (PPO), allocates Physical Resource Blocks (PRBs) across the three canonical 5G network slices
i.e. eMBB, mMTC, and URLLC; deciding the number of PRBs to keep active in order to balance QoS and energy efficiency

The agent is evaluated against three classical schedulers Round Robin, Max-Throughput and Proportional Fair; and a Best stastic benchmark over three traffic scenarios and results interactively visualized in Plotly Dash Dashboard

Research question: To what degree can a single-agent RL scheduler trained using PPO outperform traditional algorithms in optimising both energy efficiency and Quality of Service across heterogeneous 5G slices?

# Repository structure
```
5G_RL-Based_Scheduler/
├── environment/
│   ├── env_5g.py          # Gymnasium environment: slices, reward, traffic scenarios, power model
│   └── test_env.py        # environment validation tests
├── agents/
│   └── train_ppo.py       # Trains the PPO agent
├── baselines/
│   └── schedulers.py      # Round Robin, Max-Throughput, Proportional Fair, Best Static
├── evaluation/
│   └── evaluate.py        # Compares all schedulers; t-tests and confidence intervals 
├── dashboard/
│   └── app.py             # Plotly Dash performance dashboard
├── results/
│   └── v2/                # Trained models, training logs and evaluation results
│       ├── main/seed_42/  # One folder per training seed (models, logs, config.json)
│       ├── main/seed_43/
│       ├── main/seed_44/
│       ├── evaluation_results.csv
│       ├── evaluation_episodes.csv
│       ├── ppo_seed_summary.csv
│       └── significance_tests.csv
├── get_results.py         # Prints training-curve and PRB allocations values
├── requirements.txt       # Exact package versions used

```
---

## Requirements

- **Python 3.13** (developed and tested on 3.13)
- **Operating system:** Windows, macOS or Linux
- **Hardware:** standard laptop CPU
- **Disk space:** about 2 GB for the virtual environment (mainly PyTorch)

# Main libraries 
| Library | Purpose |
|---|---|
| Gymnasium | Simulation environment framework |
| Stable-Baselines3 | PPO implementation and training callbacks |
| PyTorch | Neural network backend |
| NumPy | Seeded traffic generation and Numerical computation |
| Pandas | CSV export and Results tables |
| Plotly and Dash | Interactive dashboard |
| TensorBoard | Training logs |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Anita-J-Kimutai/Project.git
cd Project
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```
You should see `(venv)` at the start of the terminal line.

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

---

## Running the project

> **Important:** all the command should be run from the **project root**, and run scripts as **modules** with `python -m` and dots, not file paths.
>
> ✔ `python -m dashboard.app` ✘ `python dashboard/app.py`


### Step 1: Test the environment

```bash
python -m environment.test_env
```
Expected: four `[PASS]` lines, a results table, and `All environment tests passed.`

### Step 2: Test the baseline schedulers

```bash
python -m baselines.schedulers
```
Expected: tabular results per scenario

### Step 3: Train the PPO agent 

```bash
python -m agents.train_ppo
```
This trains three agents (seeds 42, 43 and 44) for 500,000 steps each and saves them in `results/v2/main/seed_XX/`.

Optional arguments:

### Step 4: Evaluate all schedulers (about 5–12 mins)

```bash
python -m evaluation.evaluate
```
Runs PPO and the baselines for 100 episodes per scenario on similar traffic, then prints the results with significance tests and 95% confidence intervals.

### Step 5: Launch the dashboard

```bash
python -m dashboard.app
```
Open **http://127.0.0.1:8050** in a browser. Use the dropdowns to switch performance metric and traffic scenario. To stop the dashboard, press `Ctrl + C` in the terminal.

The dashboard has six panels: scheduler comparison, QoS–energy trade-off, significance tests, training curve, PRB allocation heatmap and summary statistics.


## Configurations

Key settings in `environment/env_5g.py`:

- **Traffic scenarios** (`SCENARIOS`): mean arrivals per slot for each slice in Balanced, URLLC-Critical and eMBB-Dominant.
- **Action space**: 5 PRB levels (60–100) ×  10 split patterns  = 50 actions
- **Power model**: 186.4 W at 60 PRBs to 224 W at 100 PRBs: base station power from (Auer et al., 2011)
- **Reward weights**: default w1 = 0.4 (eMBB), w2 = 0.3 (URLLC), w3 = 0.2 (mMTC), w4 = 0.1 (energy)

PPO hyperparameters are set in `agents/train_ppo.py`.

---

## Reproducibility

- All traffic is generated from seeded random number generators, the same seed will always produce the same traffic.
- PPO is trained with seeds 42, 43 and 44.

- Configurations are saved in `config.json`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'environment'` | Run from the project root, using `python -m folder.file` |
| `ModuleNotFoundError: No module named 'gymnasium'` (or another library) | Activate the virtual environment, then `pip install -r requirements.txt` |
| Dashboard panel says "No training logs found" or "No evaluation results found" | First run `python -m agents.train_ppo` and `python -m evaluation.evaluate`  |


---

