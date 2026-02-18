"""
VPS Validation - Module 2: Inter-Rater Reliability
===================================================
Computes agreement statistics between two or more raters
before trusting the human labels as "ground truth."

Metrics:
  - Cohen's Kappa        (two raters, ordinal/categorical)
  - Krippendorff's Alpha (2+ raters, ordinal scale)
  - Spearman correlation (pairwise, per skill)
  - Mean Absolute Difference (intuitive magnitude check)

Usage:
    python 2_inter_rater_reliability.py \
        --r1 annotations/annotations_Alice_*.csv \
        --r2 annotations/annotations_Bob_*.csv

    # Or pass a directory containing all rater CSVs:
    python 2_inter_rater_reliability.py --dir annotations/
"""

import argparse
import glob
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

SKILLS = ["Respect", "Support", "Education", "Planning", "Engagement", "Communication"]

KAPPA_BENCHMARKS = [
    (0.00, "No agreement (chance level)"),
    (0.20, "Slight agreement"),
    (0.40, "Fair agreement"),
    (0.60, "Moderate agreement  ← acceptable minimum"),
    (0.80, "Substantial agreement  ← good for research"),
    (1.00, "Perfect agreement"),
]


# ── Loading ───────────────────────────────────────────────────────────────────

def load_rater_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Infer rater name from column if missing in args
    if "rater" not in df.columns:
        df["rater"] = os.path.basename(path).split("_")[1]
    return df[["conv_id", "rater"] + SKILLS]


def load_from_dir(directory: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(directory, "annotations_*.csv"))
    if not files:
        raise FileNotFoundError(f"No annotation CSVs found in {directory}")
    return pd.concat([load_rater_file(f) for f in files], ignore_index=True)


def pivot_raters(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Returns a dict: skill -> DataFrame with conv_id as index, raters as columns.
    Only includes conversations rated by ALL raters (inner join).
    """
    raters = df["rater"].unique()
    pivoted = {}
    for skill in SKILLS:
        skill_df = df.pivot(index="conv_id", columns="rater", values=skill)
        skill_df = skill_df.dropna()  # keep only conversations rated by everyone
        pivoted[skill] = skill_df
    return pivoted, raters


# ── Cohen's Kappa ─────────────────────────────────────────────────────────────

def cohen_kappa(r1_scores: np.ndarray, r2_scores: np.ndarray, weights: str = "linear") -> float:
    """
    Weighted Cohen's Kappa. Linear weights penalize larger disagreements more.
    """
    from sklearn.metrics import cohen_kappa_score  # soft dependency
    try:
        return cohen_kappa_score(r1_scores, r2_scores, weights=weights)
    except Exception:
        # Manual fallback if sklearn not available
        n = len(r1_scores)
        cats = sorted(set(r1_scores) | set(r2_scores))
        k = len(cats)
        cat_idx = {c: i for i, c in enumerate(cats)}

        # Observed agreement matrix
        O = np.zeros((k, k))
        for a, b in zip(r1_scores, r2_scores):
            O[cat_idx[a], cat_idx[b]] += 1
        O /= n

        # Expected agreement
        row_marginals = O.sum(axis=1)
        col_marginals = O.sum(axis=0)
        E = np.outer(row_marginals, col_marginals)

        # Linear weights
        W = np.array([[1 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)])

        Po = (W * O).sum()
        Pe = (W * E).sum()
        return (Po - Pe) / (1 - Pe) if (1 - Pe) != 0 else 1.0


# ── Krippendorff's Alpha ──────────────────────────────────────────────────────

def krippendorff_alpha(data: np.ndarray, level: str = "ordinal") -> float:
    """
    data: shape (n_raters, n_items). NaN = missing rating.
    level: 'ordinal' | 'interval' | 'nominal'
    """
    data = np.array(data, dtype=float)
    n_raters, n_items = data.shape

    # Coincidence matrix
    values = sorted(set(data[~np.isnan(data)]))
    v = {val: i for i, val in enumerate(values)}
    m = len(values)

    coincidences = np.zeros((m, m))
    for item in range(n_items):
        col = data[:, item]
        obs = col[~np.isnan(col)]
        n_obs = len(obs)
        if n_obs < 2:
            continue
        for i in range(len(obs)):
            for j in range(len(obs)):
                if i != j:
                    coincidences[v[obs[i]], v[obs[j]]] += 1 / (n_obs - 1)

    # Metric function
    if level == "ordinal":
        g = np.zeros((m, m))
        nc = coincidences.sum(axis=1)  # row sums (n_c)
        for c in range(m):
            for k in range(m):
                # Sum from c+1 to k (inclusive) of n_g, plus half of n_c and n_k
                inner = sum(nc[g_] for g_ in range(c + 1, k)) if k > c + 1 else 0
                g[c, k] = (inner + nc[c] / 2 + nc[k] / 2) ** 2
                g[k, c] = g[c, k]
        metric = g
    elif level == "interval":
        metric = np.array([[(values[c] - values[k]) ** 2 for k in range(m)] for c in range(m)])
    else:  # nominal
        metric = 1 - np.eye(m)

    n = coincidences.sum()
    nc = coincidences.sum(axis=1)

    D_o = (coincidences * metric).sum() / n
    D_e = (np.outer(nc, nc) * metric).sum() / (n * (n - 1))

    return 1 - D_o / D_e if D_e != 0 else 1.0


# ── Reporting ─────────────────────────────────────────────────────────────────

def interpret_kappa(k: float) -> str:
    for threshold, label in reversed(KAPPA_BENCHMARKS):
        if k >= threshold:
            return label
    return "Below chance"


def run_analysis(df: pd.DataFrame):
    pivoted, raters = pivot_raters(df)
    rater_pairs = list(combinations(raters, 2))

    print("\n" + "=" * 65)
    print("  VPS INTER-RATER RELIABILITY REPORT")
    print("=" * 65)
    print(f"\n  Raters  : {', '.join(raters)}")
    print(f"  Skills  : {', '.join(SKILLS)}")
    n_shared = min(len(pivoted[s]) for s in SKILLS)
    print(f"  Shared conversations (rated by all): {n_shared}")

    # ── Per-skill summary ──────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  PER-SKILL SUMMARY")
    print("-" * 65)

    skill_results = []
    for skill in SKILLS:
        sdf = pivoted[skill]
        if sdf.empty:
            continue

        # Krippendorff's alpha (works for 2+ raters)
        alpha = krippendorff_alpha(sdf.values.T, level="ordinal")

        # Mean absolute difference (pairwise average)
        mad_vals = []
        kappa_vals = []
        rho_vals = []
        for r1, r2 in rater_pairs:
            if r1 in sdf.columns and r2 in sdf.columns:
                a, b = sdf[r1].values, sdf[r2].values
                mad_vals.append(np.mean(np.abs(a - b)))
                kappa_vals.append(cohen_kappa(a.astype(int), b.astype(int)))
                rho, _ = stats.spearmanr(a, b)
                rho_vals.append(rho)

        row = {
            "Skill": skill,
            "Kripp. α": f"{alpha:.3f}",
            "Cohen's κ (avg)": f"{np.mean(kappa_vals):.3f}" if kappa_vals else "N/A",
            "Spearman ρ (avg)": f"{np.mean(rho_vals):.3f}" if rho_vals else "N/A",
            "MAD (avg)": f"{np.mean(mad_vals):.2f}" if mad_vals else "N/A",
            "Interpretation": interpret_kappa(alpha),
        }
        skill_results.append(row)

    results_df = pd.DataFrame(skill_results).set_index("Skill")
    print(results_df.to_string())

    # ── Pairwise detail ────────────────────────────────────────────────────────
    if len(rater_pairs) > 0:
        print("\n" + "-" * 65)
        print("  PAIRWISE KAPPA DETAIL (Linear-Weighted Cohen's κ)")
        print("-" * 65)

        for r1, r2 in rater_pairs:
            print(f"\n  {r1} vs {r2}:")
            for skill in SKILLS:
                sdf = pivoted[skill]
                if r1 in sdf.columns and r2 in sdf.columns and not sdf.empty:
                    k = cohen_kappa(sdf[r1].values.astype(int), sdf[r2].values.astype(int))
                    bar = "█" * int(max(0, k) * 20)
                    print(f"    {skill:<15} κ = {k:+.3f}  {bar}")

    # ── Overall verdict ────────────────────────────────────────────────────────
    alphas = [float(r["Kripp. α"]) for r in skill_results]
    mean_alpha = np.mean(alphas)
    print("\n" + "-" * 65)
    print(f"  OVERALL  Krippendorff's α (mean across skills): {mean_alpha:.3f}")
    print(f"  → {interpret_kappa(mean_alpha)}")

    if mean_alpha >= 0.60:
        verdict = "✅  Agreement is sufficient. Human labels can serve as ground truth."
    elif mean_alpha >= 0.40:
        verdict = "⚠️   Fair agreement. Consider resolving disagreements via discussion before using labels."
    else:
        verdict = "❌  Poor agreement. Raters need calibration before labels are trusted."
    print(f"\n  {verdict}\n")

    return results_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VPS Inter-Rater Reliability Analysis")
    parser.add_argument("--r1",  default=None, help="Annotation CSV for rater 1")
    parser.add_argument("--r2",  default=None, help="Annotation CSV for rater 2")
    parser.add_argument("--dir", default=None, help="Directory containing all rater CSVs")
    args = parser.parse_args()

    if args.dir:
        df = load_from_dir(args.dir)
    elif args.r1 and args.r2:
        df = pd.concat([load_rater_file(args.r1), load_rater_file(args.r2)], ignore_index=True)
    else:
        # Demo with synthetic data
        print("\n  ℹ  No files specified — running demo with synthetic rater data.\n")
        np.random.seed(42)
        conv_ids = [f"conv_{i:03d}" for i in range(20)]
        rows = []
        for rater, noise in [("Alice", 0), ("Bob", 0.5), ("Carol", 0.8)]:
            for cid in conv_ids:
                true_score = np.random.randint(1, 6)
                row = {"conv_id": cid, "rater": rater}
                for skill in SKILLS:
                    score = int(np.clip(true_score + np.random.choice([-noise, 0, noise]), 1, 5))
                    row[skill] = score
                rows.append(row)
        df = pd.DataFrame(rows)

    run_analysis(df)


if __name__ == "__main__":
    main()
