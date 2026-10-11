"""
REINFORCEMENT LEARNING-BASED QUALITY OF SERVICE-AWARE
RESOURCE SCHEDULING IN 5G NETWORK SLICING

PLOTLY DASH PERFORMANCE DASHBOARD 
Description:
    Interactive Plotly Dash dashboard visualises the comparative
    performance of the PPO agent against four baseline schedulers
    across three 5G network slicing traffic scenarios.

    Panels:
        1. PPO training curve (all seeds, from real training logs)
        2. Scheduler comparison bar chart (mean +/- 95% CI)
        3. PRB allocation heatmap (measured mean PRBs per slice)
        4. Summary statistics table
        5. QoS-energy trade-off (reward vs. base station power)
        6. Statistical significance tests

    Access at : http://127.0.0.1:8050
"""

import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

# CONFIGURATION
"""
Build paths from this file's location, so the dashboard finds the
results no matter which folder it is started from.
"""
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "v2")
RESULTS_PATH = os.path.join(RESULTS_DIR, "evaluation_results.csv")
SIGNIFICANCE_PATH = os.path.join(RESULTS_DIR, "significance_tests.csv")
# "*" is a wildcard: matches seed_42, seed_43, seed_44 ...
EVAL_LOG_PATTERN = os.path.join(RESULTS_DIR, "main", "seed_*", "logs", "evaluations.npz")

TOTAL_PRBS = 100

# Colour scheme maintained throughout
COLOURS = {
    "PPO"               : "#4A90D9",
    "Round Robin"       : "#BA7517",
    "Proportional Fair" : "#1E8B4C",
    "Max-Throughput"    : "#B03030",
    "Best Static"       : "#8B949E",
    "header"            : "#1F4E79",
    "background"        : "#0D1117",
    "card"              : "#161B22",
    "text"              : "#E6EDF3",
    "muted"             : "#8B949E",
    "border"            : "#30363D",
}

SCHEDULERS = ["PPO", "Round Robin", "Proportional Fair", "Max-Throughput", "Best Static"]
SCENARIOS = ["Balanced", "eMBB-Dominant", "URLLC-Critical"]

# Dropdown options for different metrics

METRICS = {
    "reward"               : "Mean Episode Reward",
    "embb_throughput"      : "eMBB Throughput (Mbps)",
    "embb_satisfaction"    : "eMBB Satisfaction (0-1)",
    "urllc_latency"        : "URLLC Latency (ms)",
    "urllc_violation_rate" : "URLLC SLA Violation Rate",
    "mmtc_packet_loss"     : "mMTC Packet Loss Rate",
    "power_watts"          : "Base Station Power (W)",
    "energy_efficiency"    : "Energy Efficiency (Mbit/J)",
    "active_prbs"          : "Active PRBs",
    "jain_fairness"        : "Jain's Fairness Index",
}

# Shared chart styling (applied to every figure)
BASE_LAYOUT = dict(
    paper_bgcolor=COLOURS["card"],
    plot_bgcolor=COLOURS["background"],
    font=dict(color=COLOURS["text"]),
    margin=dict(l=60, r=20, t=70, b=50),
)

GRAPH_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "pan2d"],
    "displaylogo": False,
}

# LOAD DATA
def load_csv(path, label):
    #Load a CSV file, or return an empty table with a warning.
    if not os.path.exists(path):
        print(f"  WARNING: {label} not found: {path}")
        print("           Run: python -m evaluation.evaluate")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from {os.path.relpath(path, PROJECT_ROOT)}")
    return df


def load_training_curves():
    """
    Load the training-time evaluation curves for every seed.

    EvalCallback (in train_ppo.py) tested the agent every 10,000 steps
    and saved the results in evaluations.npz:
        timesteps - when each test happened
        results - rewards of each test episode 
    These are cleaner than raw training episodes, because training
    episodes mix random scenarios and exploration noise.

    Returns a list of (seed, timesteps, mean rewards); empty if none found.
    """
    curves = []
    for path in sorted(glob.glob(EVAL_LOG_PATTERN)):
        # Folder name is "seed_42" - take the number after "_"
        seed = os.path.basename(os.path.dirname(os.path.dirname(path))).split("_")[-1]
        data = np.load(path)
        curves.append((seed, data["timesteps"], data["results"].mean(axis=1)))
    return curves

# CHART BUILDERS
def empty_figure(message):
    """A blank chart showing a message (used when data is missing)."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(size=14, color=COLOURS["muted"]))
    fig.update_layout(**BASE_LAYOUT)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def build_training_curve():
    """Line chart of evaluation reward during training, one line per seed."""
    curves = load_training_curves()
    if not curves:
        return empty_figure("No training logs found.<br>Run: python -m agents.train_ppo")

    fig = go.Figure()

    # Thin line for each seed
    for seed, steps, rewards in curves:
        fig.add_trace(go.Scatter(
            x=steps, y=rewards, mode="lines", name=f"Seed {seed}",
            line=dict(width=1), opacity=0.5,
        ))

    # Thick line for the mean across seeds (only where all seeds have data)
    min_len = min(len(r) for _, _, r in curves)
    mean_rewards = np.mean([r[:min_len] for _, _, r in curves], axis=0)
    fig.add_trace(go.Scatter(
        x=curves[0][1][:min_len], y=mean_rewards, mode="lines",
        name="Mean of seeds", line=dict(color=COLOURS["PPO"], width=3),
    ))

    fig.update_layout(
        title="PPO Training Curve (evaluation every 10k steps)",
        xaxis_title="Training timesteps",
        yaxis_title="Mean episode reward",
        legend=dict(orientation="h", y=-0.25),
        **BASE_LAYOUT,
    )
    fig.update_xaxes(gridcolor=COLOURS["border"])
    fig.update_yaxes(gridcolor=COLOURS["border"])
    return fig


def build_comparison_chart(df, scenario, metric):
    #Bar chart comparing all schedulers; error bars = 95% CI.
    if df.empty:
        return empty_figure("No evaluation results found.<br>Run: python -m evaluation.evaluate")

    filtered = df[df["scenario"] == scenario]
    fig = go.Figure()

    for scheduler in SCHEDULERS:
        row = filtered[filtered["scheduler"] == scheduler]
        if row.empty:
            continue
        mean_val = row[f"mean_{metric}"].values[0]
        ci_val = row[f"ci95_{metric}"].values[0]

        fig.add_trace(go.Bar(
            name=scheduler,
            x=[scheduler],
            y=[mean_val],
            error_y=dict(type="data", array=[ci_val], visible=True),
            marker_color=COLOURS[scheduler],
            text=[f"{mean_val:.3f}"],
            textposition="outside",
            cliponaxis=False,    # stops the value labels being cut off
        ))

    label = METRICS.get(metric, metric)
    fig.update_layout(
        title=f"{label} — {scenario}",
        yaxis_title=label,
        showlegend=False,
        **BASE_LAYOUT,
    )
    fig.update_xaxes(gridcolor=COLOURS["border"])
    fig.update_yaxes(gridcolor=COLOURS["border"])
    return fig


def build_heatmap(df, scenario):
    """
    Heatmap of the measured mean PRBs each scheduler gave each slice,
    plus the PRBs left inactive 
    """
    if df.empty or "mean_embb_prbs" not in df.columns:
        return empty_figure("No PRB allocation data.<br>Re-run: python -m evaluation.evaluate")

    filtered = df[df["scenario"] == scenario]
    rows, names = [], []
    for scheduler in SCHEDULERS:
        row = filtered[filtered["scheduler"] == scheduler]
        if row.empty:
            continue
        embb = row["mean_embb_prbs"].values[0]
        urllc = row["mean_urllc_prbs"].values[0]
        mmtc = row["mean_mmtc_prbs"].values[0]
        inactive = TOTAL_PRBS - (embb + urllc + mmtc)
        rows.append([embb, urllc, mmtc, inactive])
        names.append(scheduler)

    z = np.array(rows)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["eMBB", "URLLC", "mMTC", "Inactive"],
        y=names,
        colorscale="Blues",
        zmin=0, zmax=TOTAL_PRBS,
        text=np.round(z, 1),
        texttemplate="%{text}",
        colorbar=dict(title="PRBs"),
    ))
    fig.update_layout(
        title=f"Mean PRB Allocation per Slice — {scenario}",
        xaxis_title="Network slice",
        **BASE_LAYOUT,
    )
    # Reverse the y axis so PPO appears at the top
    fig.update_yaxes(autorange="reversed")
    return fig


def build_tradeoff_chart(df, scenario):
    """
    Scatter: base station power (x) vs reward (y) for each scheduler.
    Top-left is best: high QoS reward for low power.
    Error bars show 95% CIs on both axes.
    """
    if df.empty:
        return empty_figure("No evaluation results found.")

    filtered = df[df["scenario"] == scenario]
    fig = go.Figure()
    for scheduler in SCHEDULERS:
        row = filtered[filtered["scheduler"] == scheduler]
        if row.empty:
            continue
        fig.add_trace(go.Scatter(
            x=row["mean_power_watts"], y=row["mean_reward"],
            error_x=dict(type="data", array=row["ci95_power_watts"]),
            error_y=dict(type="data", array=row["ci95_reward"]),
            mode="markers+text", name=scheduler,
            text=[scheduler], textposition="top center",
            marker=dict(size=14, color=COLOURS[scheduler]),
        ))

    fig.update_layout(
        title=f"QoS–Energy Trade-off — {scenario} (top-left is best)",
        xaxis_title="Mean base station power (W)",
        yaxis_title="Mean episode reward",
        showlegend=False,
        **BASE_LAYOUT,
    )
    fig.update_xaxes(gridcolor=COLOURS["border"])
    fig.update_yaxes(gridcolor=COLOURS["border"])
    return fig


# Shared styling for both tables
TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_header={
        "backgroundColor": COLOURS["header"],
        "color": COLOURS["text"],
        "fontWeight": "bold",
        "textAlign": "center",
        "border": f"1px solid {COLOURS['border']}",
        "whiteSpace": "normal",
    },
    style_cell={
        "backgroundColor": COLOURS["card"],
        "color": COLOURS["text"],
        "textAlign": "center",
        "border": f"1px solid {COLOURS['border']}",
        "padding": "8px",
        "fontFamily": "Arial",
        "fontSize": "12px",
    },
)


def build_summary_table(df, scenario):
    #Summary statistics table for one scenario (PPO row highlighted)
    if df.empty:
        return html.P("No evaluation results found.", style={"color": COLOURS["muted"]})

    filtered = df[df["scenario"] == scenario].copy()

    # Reward shown as "mean ± CI"; other metrics as means
    filtered["Reward (±95% CI)"] = filtered.apply(
        lambda r: f"{r['mean_reward']:.2f} ± {r['ci95_reward']:.2f}", axis=1)

    table = pd.DataFrame({
        "Scheduler": filtered["scheduler"],
        "Reward (±95% CI)": filtered["Reward (±95% CI)"],
        "eMBB Sat.": filtered["mean_embb_satisfaction"].round(3),
        "URLLC Lat. (ms)": filtered["mean_urllc_latency"].round(2),
        "URLLC Viol.": filtered["mean_urllc_violation_rate"].round(3),
        "mMTC Loss": filtered["mean_mmtc_packet_loss"].round(3),
        "Power (W)": filtered["mean_power_watts"].round(1),
        "Mbit/J": filtered["mean_energy_efficiency"].round(3),
        "Jain": filtered["mean_jain_fairness"].round(3),
    })

    return dash_table.DataTable(
        data=table.to_dict("records"),
        columns=[{"name": c, "id": c} for c in table.columns],
        style_data_conditional=[{
            # Highlight the PPO row wherever it appears
            "if": {"filter_query": '{Scheduler} = "PPO"'},
            "backgroundColor": "rgba(74,144,217,0.25)",
            "fontWeight": "bold",
        }],
        **TABLE_STYLE,
    )


def build_significance_table(sig, scenario):
    #Welch t-test results: PPO vs best classical and vs Best Static.
    if sig.empty:
        return html.P("No significance test results found.",
                      style={"color": COLOURS["muted"]})

    filtered = sig[sig["scenario"] == scenario]
    table = pd.DataFrame({
        "Comparison": "PPO vs " + filtered["reference"],
        "Difference": filtered["difference"].map(lambda d: f"{d:+.2f}"),
        "p value": filtered["p_value"].map(lambda p: "< 0.0001" if p < 0.0001 else f"{p:.4f}"),
        "Result": filtered["significant_at_0.05"].map(
            lambda s: "Significant" if s else "Not significant"),
    })

    return dash_table.DataTable(
        data=table.to_dict("records"),
        columns=[{"name": c, "id": c} for c in table.columns],
        **TABLE_STYLE,
    )

# DASH APP LAYOUT
print("\n  Loading dashboard data ...")
df_results = load_csv(RESULTS_PATH, "Evaluation results")
df_significance = load_csv(SIGNIFICANCE_PATH, "Significance tests")

app = dash.Dash(
    __name__,
    title="5G RL Scheduler Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)


def card(children, padding="10px"):
    #Reusable dark panel used for every chart and table
    return html.Div(
        style={
            "backgroundColor": COLOURS["card"],
            "border": f"1px solid {COLOURS['border']}",
            "borderRadius": "8px",
            "padding": padding,
        },
        children=children,
    )


def row(children):
    #Two-column grid row.
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
               "gap": "20px", "marginBottom": "20px"},
        children=children,
    )


def section_title(text):
    return html.H3(text, style={"color": COLOURS["text"], "marginTop": 0,
                                "marginBottom": "15px", "fontSize": "16px"})


def note(text):
    return html.P(text, style={"color": COLOURS["muted"], "fontSize": "11px",
                               "marginTop": "10px"})


LABEL_STYLE = {"color": COLOURS["text"], "fontWeight": "bold",
               "marginBottom": "5px", "display": "block"}

header = html.Div(
    style={"backgroundColor": COLOURS["header"], "padding": "20px 30px",
           "marginBottom": "20px"},
    children=[
        html.H1("5G Network Slicing — RL Scheduler Dashboard",
                style={"color": "white", "margin": 0, "fontSize": "24px"}),
        html.P("Reinforcement Learning-Based QoS-Aware and Energy-Aware "
               "Resource Scheduling in 5G Network Slicing",
               style={"color": "#AACCEE", "margin": "5px 0 0 0", "fontSize": "13px"}),
    ],
)

# Legend chips built in a loop (one coloured square per scheduler)
legend = html.Div(
    style={"marginLeft": "auto", "display": "flex", "gap": "15px",
           "alignItems": "center", "flexWrap": "wrap"},
    children=[
        html.Div([
            html.Span("■ ", style={"color": COLOURS[s]}),
            html.Span(s, style={"color": COLOURS["text"], "fontSize": "13px"}),
        ])
        for s in SCHEDULERS
    ],
)

controls = html.Div(
    style={"backgroundColor": COLOURS["card"], "padding": "15px 30px",
           "marginBottom": "20px", "border": f"1px solid {COLOURS['border']}",
           "display": "flex", "gap": "30px", "alignItems": "center",
           "flexWrap": "wrap"},
    children=[
        html.Div([
            html.Label("Traffic Scenario:", style=LABEL_STYLE),
            dcc.Dropdown(
                id="scenario-dropdown",
                options=[{"label": s, "value": s} for s in SCENARIOS],
                value="Balanced", clearable=False,
                style={"width": "220px", "color": "black"},
            ),
        ]),
        html.Div([
            html.Label("Performance Metric:", style=LABEL_STYLE),
            dcc.Dropdown(
                id="metric-dropdown",
                options=[{"label": v, "value": k} for k, v in METRICS.items()],
                value="reward", clearable=False,
                style={"width": "280px", "color": "black"},
            ),
        ]),
        legend,
    ],
)

app.layout = html.Div(
    style={"backgroundColor": COLOURS["background"], "minHeight": "100vh",
           "fontFamily": "Arial, sans-serif"},
    children=[
        header,
        html.Div(style={"padding": "0 20px 20px 20px"}, children=[
            controls,

            # Row 1: training curve + scheduler comparison
            row([
                card(dcc.Graph(id="training-curve", figure=build_training_curve(),
                               config=GRAPH_CONFIG, style={"height": "380px"})),
                card(dcc.Graph(id="comparison-chart", config=GRAPH_CONFIG,
                               style={"height": "380px"})),
            ]),

            # Row 2: PRB heatmap + QoS-energy trade-off
            row([
                card(dcc.Graph(id="heatmap", config=GRAPH_CONFIG,
                               style={"height": "380px"})),
                card(dcc.Graph(id="tradeoff-chart", config=GRAPH_CONFIG,
                               style={"height": "380px"})),
            ]),

            # Row 3: summary table + significance tests
            row([
                card([
                    section_title("Summary Statistics"),
                    html.Div(id="summary-table"),
                    note("PPO highlighted. PPO values are pooled over 3 training "
                         "seeds x 100 evaluation episodes; baselines over 100 "
                         "episodes. All schedulers face identical traffic."),
                ], padding="20px"),
                card([
                    section_title("Statistical Significance (Welch t-test)"),
                    html.Div(id="significance-table"),
                    note("Best Static is an oracle benchmark: it is chosen knowing "
                         "the scenario in advance, which PPO is never told. "
                         "Read p values together with the size of the difference."),
                ], padding="20px"),
            ]),
        ]),
    ],
)

# CALLBACKS

# Each callback re-draws one panel whenever a dropdown changes.
@app.callback(
    Output("comparison-chart", "figure"),
    [Input("scenario-dropdown", "value"), Input("metric-dropdown", "value")],
)
def update_comparison(scenario, metric):
    return build_comparison_chart(df_results, scenario, metric)


@app.callback(Output("heatmap", "figure"), Input("scenario-dropdown", "value"))
def update_heatmap(scenario):
    return build_heatmap(df_results, scenario)


@app.callback(Output("tradeoff-chart", "figure"), Input("scenario-dropdown", "value"))
def update_tradeoff(scenario):
    return build_tradeoff_chart(df_results, scenario)


@app.callback(Output("summary-table", "children"), Input("scenario-dropdown", "value"))
def update_table(scenario):
    return build_summary_table(df_results, scenario)


@app.callback(Output("significance-table", "children"), Input("scenario-dropdown", "value"))
def update_significance(scenario):
    return build_significance_table(df_significance, scenario)

# ENTRY POINT

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  5G RL SCHEDULER — PERFORMANCE DASHBOARD")
    print("=" * 55)
    print("  Open browser at: http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop")
    print("=" * 55 + "\n")
    app.run(debug=True, host="127.0.0.1", port=8050)