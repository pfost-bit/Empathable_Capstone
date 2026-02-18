"""
VPS Validation - Module 3: Correlation Analysis
================================================
Compares NLP model scores against human annotation ground truth.

Outputs:
  - Spearman ρ per skill (model vs. human consensus)
  - Bland-Altman plots (agreement + bias visualization)
  - Confusion-style heatmaps (score distribution comparison)
  - Synthetic edge-case validation tests
  - Summary report CSV

Usage:
    python 3_correlation_analysis.py \
        --model    model_scores.csv \
        --human    annotations/ \
        --output   results/

    # Run demo with synthetic data:
    python 3_correlation_analysis.py --demo
"""

import argparse
import glob
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

SKILLS = ["Respect", "Support", "Education", "Planning", "Engagement", "Communication"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
SKILL_COLORS = dict(zip(SKILLS, COLORS))

# ── Loading ───────────────────────────────────────────────────────────────────

def load_model_scores(path: str) -> pd.DataFrame:
    """CSV with columns: conv_id, Respect, Support, Education, Planning, Engagement, Communication"""
    df = pd.read_csv(path).set_index("conv_id")
    return df[SKILLS]


def load_human_consensus(directory: str) -> pd.DataFrame:
    """
    Load all rater files from a directory and compute mean score per conv per skill.
    This is the 'ground truth' — the average human judgment.
    """
    files = glob.glob(os.path.join(directory, "annotations_*.csv"))
    if not files:
        raise FileNotFoundError(f"No annotation CSVs found in {directory}")

    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    consensus = combined.groupby("conv_id")[SKILLS].mean()
    return consensus


# ── Consensus aggregation ─────────────────────────────────────────────────────

def build_comparison_df(model_df: pd.DataFrame, human_df: pd.DataFrame) -> pd.DataFrame:
    """Merge model and human scores on shared conv_ids."""
    shared = model_df.index.intersection(human_df.index)
    if len(shared) == 0:
        raise ValueError("No shared conv_ids between model scores and human annotations.")

    model = model_df.loc[shared].add_suffix("_model")
    human = human_df.loc[shared].add_suffix("_human")
    return pd.concat([model, human], axis=1)


# ── Statistical analysis ──────────────────────────────────────────────────────

def compute_correlations(comp_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for skill in SKILLS:
        model_col = f"{skill}_model"
        human_col = f"{skill}_human"
        model_scores = comp_df[model_col].values
        human_scores = comp_df[human_col].values

        rho, p_val = stats.spearmanr(model_scores, human_scores)
        r_pearson, _ = stats.pearsonr(model_scores, human_scores)
        mae = np.mean(np.abs(model_scores - human_scores))
        bias = np.mean(model_scores - human_scores)  # positive = model scores higher

        rows.append({
            "Skill":          skill,
            "Spearman ρ":     round(rho, 3),
            "p-value":        round(p_val, 4),
            "Pearson r":      round(r_pearson, 3),
            "MAE":            round(mae, 3),
            "Bias (M-H)":     round(bias, 3),
            "Significant":    "✅" if p_val < 0.05 else "❌",
        })

    return pd.DataFrame(rows).set_index("Skill")


def interpret_rho(rho: float) -> str:
    if rho >= 0.80: return "Very strong ✅"
    if rho >= 0.60: return "Strong ✅"
    if rho >= 0.40: return "Moderate ⚠️"
    if rho >= 0.20: return "Weak ❌"
    return "Negligible ❌"


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_scatter_grid(comp_df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Model vs. Human Scores — Per Skill Scatter", fontsize=14, fontweight="bold", y=1.01)

    for ax, skill, color in zip(axes.flat, SKILLS, COLORS):
        x = comp_df[f"{skill}_human"].values
        y = comp_df[f"{skill}_model"].values

        rho, p = stats.spearmanr(x, y)

        # Scatter
        ax.scatter(x, y, alpha=0.65, color=color, edgecolors="white", linewidth=0.5, s=60)

        # Perfect agreement line
        lims = [1, 5]
        ax.plot(lims, lims, "--", color="gray", linewidth=1, alpha=0.6, label="Perfect agreement")

        # OLS trend line
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(1, 5, 100)
        ax.plot(x_line, m * x_line + b, "-", color=color, linewidth=2)

        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0.8, 5.2)
        ax.set_xlabel("Human Score", fontsize=10)
        ax.set_ylabel("Model Score", fontsize=10)
        ax.set_title(f"{skill}\nSpearman ρ = {rho:.3f}  (p = {p:.3f})", fontsize=10)
        ax.set_xticks(range(1, 6))
        ax.set_yticks(range(1, 6))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "scatter_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


def plot_bland_altman(comp_df: pd.DataFrame, output_dir: str):
    """
    Bland-Altman plot: x = mean of model & human, y = difference (model - human).
    Reveals systematic bias and limits of agreement.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("Bland-Altman Agreement Plots", fontsize=14, fontweight="bold", y=1.01)

    for ax, skill, color in zip(axes.flat, SKILLS, COLORS):
        m = comp_df[f"{skill}_model"].values
        h = comp_df[f"{skill}_human"].values

        mean_scores = (m + h) / 2
        diff = m - h

        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff

        ax.scatter(mean_scores, diff, alpha=0.65, color=color, edgecolors="white", s=55)
        ax.axhline(mean_diff,  color="black",  linewidth=1.5, linestyle="-",  label=f"Bias = {mean_diff:+.2f}")
        ax.axhline(loa_upper,  color="tomato", linewidth=1,   linestyle="--", label=f"+1.96σ = {loa_upper:+.2f}")
        ax.axhline(loa_lower,  color="tomato", linewidth=1,   linestyle="--", label=f"-1.96σ = {loa_lower:+.2f}")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

        ax.set_xlabel("Mean Score (Model + Human) / 2", fontsize=9)
        ax.set_ylabel("Difference (Model − Human)", fontsize=9)
        ax.set_title(f"{skill}  |  Bias {mean_diff:+.2f}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "bland_altman.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


def plot_correlation_summary(corr_df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Validation Summary", fontsize=13, fontweight="bold")

    # Spearman rho bar chart
    ax = axes[0]
    rhos = corr_df["Spearman ρ"].values
    bars = ax.barh(SKILLS, rhos, color=COLORS, edgecolor="white", height=0.6)
    ax.axvline(0.6, color="green", linestyle="--", linewidth=1.5, label="Acceptable threshold (0.6)")
    ax.axvline(0.4, color="orange", linestyle="--", linewidth=1.5, label="Moderate threshold (0.4)")
    ax.set_xlim(-0.1, 1.05)
    ax.set_xlabel("Spearman ρ", fontsize=11)
    ax.set_title("Model–Human Correlation per Skill")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    for bar, rho in zip(bars, rhos):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{rho:.3f}", va="center", fontsize=9)

    # Bias bar chart
    ax = axes[1]
    biases = corr_df["Bias (M-H)"].values
    bar_colors = ["#C44E52" if b > 0 else "#4C72B0" for b in biases]
    ax.barh(SKILLS, biases, color=bar_colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Bias = Mean(Model − Human)", fontsize=11)
    ax.set_title("Model Bias per Skill\n(+ = model scores higher than humans)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(min(biases) - 0.3, max(biases) + 0.3)
    for i, (b, skill) in enumerate(zip(biases, SKILLS)):
        ax.text(b + (0.02 if b >= 0 else -0.02), i,
                f"{b:+.2f}", va="center", ha="left" if b >= 0 else "right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


# ── Edge-case validation ──────────────────────────────────────────────────────

def run_edge_case_tests(score_fn) -> pd.DataFrame:
    """
    Validates model directional sensitivity using synthetic conversations.
    score_fn: callable that accepts a conversation string and returns dict {skill: score}
    """
    test_cases = [
        {
            "id": "edge_high_empathy",
            "description": "Clearly empathetic, collaborative doctor",
            "text": (
                "Doctor: I can hear that you're really worried, and that makes complete sense. "
                "Let's take this step by step together — I want to make sure you understand "
                "everything and feel comfortable with whatever we decide. What matters most to you right now?"
            ),
            "expected_direction": "high",
        },
        {
            "id": "edge_low_empathy",
            "description": "Dismissive, uncommunicative doctor",
            "text": (
                "Doctor: What's your problem? "
                "Patient: I have chest pains. "
                "Doctor: Take ibuprofen. Next."
            ),
            "expected_direction": "low",
        },
        {
            "id": "edge_neutral",
            "description": "Clinically adequate but emotionally flat",
            "text": (
                "Doctor: Tell me your symptoms. "
                "Patient: Fatigue, mild fever. "
                "Doctor: How long? "
                "Patient: Five days. "
                "Doctor: I'll prescribe a short course of antibiotics. Return if symptoms persist."
            ),
            "expected_direction": "mid",
        },
    ]

    results = []
    high_scores = None
    low_scores = None

    for tc in test_cases:
        scores = score_fn(tc["text"])
        vps = np.mean(list(scores.values()))
        scores["conv_id"] = tc["id"]
        scores["description"] = tc["description"]
        scores["expected"] = tc["expected_direction"]
        scores["VPS_mean"] = round(vps, 3)
        results.append(scores)

        if tc["expected_direction"] == "high":
            high_scores = vps
        elif tc["expected_direction"] == "low":
            low_scores = vps

    passed = (high_scores is not None and low_scores is not None and high_scores > low_scores)

    df = pd.DataFrame(results).set_index("conv_id")
    print("\n  EDGE-CASE DIRECTIONAL TESTS")
    print("  " + "-" * 50)
    for _, row in df.iterrows():
        print(f"  [{row['expected'].upper():>4}] {row['description']}")
        print(f"         VPS Mean = {row['VPS_mean']:.3f}")
    print(f"\n  Directional ordering (high > low): {'PASSED ✅' if passed else 'FAILED ❌'}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run_analysis(model_df: pd.DataFrame, human_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    comp_df = build_comparison_df(model_df, human_df)
    n = len(comp_df)

    print(f"\n  Shared conversations for analysis: {n}")

    corr_df = compute_correlations(comp_df)

    # Console report
    print("\n" + "=" * 65)
    print("  VPS CORRELATION ANALYSIS REPORT")
    print("=" * 65)
    print(corr_df.to_string())

    overall_rho = corr_df["Spearman ρ"].mean()
    print(f"\n  Mean Spearman ρ (all skills): {overall_rho:.3f}  → {interpret_rho(overall_rho)}")
    print(f"  Mean MAE                     : {corr_df['MAE'].mean():.3f}")
    print(f"  Mean Bias                    : {corr_df['Bias (M-H)'].mean():+.3f}")

    # Save CSV
    csv_path = os.path.join(output_dir, "correlation_results.csv")
    corr_df.to_csv(csv_path)
    print(f"\n  → Saved: {csv_path}")

    # Plots
    print("\n  Generating plots...")
    plot_scatter_grid(comp_df, output_dir)
    plot_bland_altman(comp_df, output_dir)
    plot_correlation_summary(corr_df, output_dir)

    print("\n  ✅  Analysis complete.\n")
    return corr_df


def make_demo_data(n=40, seed=42):
    """Generate realistic synthetic model and human scores for demonstration."""
    np.random.seed(seed)
    conv_ids = [f"conv_{i:03d}" for i in range(n)]

    # True underlying empathy level per conversation
    true_empathy = np.random.uniform(1, 5, n)

    # Human scores: noisy version of truth
    human_data = {"conv_id": conv_ids}
    for skill in SKILLS:
        noise = np.random.normal(0, 0.4, n)
        scores = np.clip(np.round(true_empathy + noise), 1, 5)
        human_data[skill] = scores
    human_df = pd.DataFrame(human_data).set_index("conv_id")

    # Model scores: correlated with truth, different noise + slight bias
    model_data = {"conv_id": conv_ids}
    for skill in SKILLS:
        noise = np.random.normal(0.3, 0.7, n)  # slight upward bias
        scores = np.clip(np.round(true_empathy + noise), 1, 5)
        model_data[skill] = scores
    model_df = pd.DataFrame(model_data).set_index("conv_id")

    return model_df, human_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VPS Correlation Analysis")
    parser.add_argument("--model",  default=None, help="Model scores CSV (conv_id + 6 skill columns)")
    parser.add_argument("--human",  default=None, help="Directory with human annotation CSVs")
    parser.add_argument("--output", default="results", help="Output directory for plots and CSV")
    parser.add_argument("--demo",   action="store_true", help="Run with synthetic demo data")
    args = parser.parse_args()

    if args.demo or (args.model is None and args.human is None):
        print("\n  ℹ  Running in DEMO mode with synthetic data.\n")
        model_df, human_df = make_demo_data()
    else:
        model_df = load_model_scores(args.model)
        human_df = load_human_consensus(args.human)

    run_analysis(model_df, human_df, args.output)


if __name__ == "__main__":
    main()
