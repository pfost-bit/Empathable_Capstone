# Empathable: Clinician Communication Scoring Platform

> An AI-powered framework for evaluating, scoring, and improving clinician-patient communication using evidence-verified, research-weighted behavioral metrics.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Sub-Criteria Formulas](#sub-criteria-formulas)
4. [VPS Scoring Framework](#vps-scoring-framework)
5. [Resiliency Scoring Framework](#resiliency-scoring-framework)
6. [Financial Model Framework](#financial-model-framework)
7. [Baseline Creation](#baseline-creation)
8. [LoRA: Patient Sub-Genre Modeling](#lora-patient-sub-genre-modeling)
9. [Visual Models](#visual-models)
10. [Visualizations](#visualizations)

---

## Project Overview

The Empathable platform uses AI to evaluate clinician communication behaviors from transcribed doctor-patient dialogues. It produces a **Value-to-Patient Score (VPS)** — a composite, research-weighted metric grounded in 130+ verified peer-reviewed publications — that quantifies how effective a clinician's communication is at driving positive patient outcomes.

The scoring architecture is layered:

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 1: SUB-CRITERIA (32 Causal Factors)                           │
│  AI evaluates clinician communication behaviors                      │
│  Each weight derived from peer-reviewed effect sizes                 │
│                            ↓                                         │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 2: 6 CORE EVALUATION SKILLS (Explanatory)                     │
│  Explains WHY the clinician received that score                      │
│  [Respect] [Support] [Education] [Planning] [Engagement] [Comms]     │
│                            ↓                                         │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 3: VALUE-TO-PATIENT SCORE (VPS)                               │
│  Calculated DIRECTLY from research-weighted sub-criteria             │
│  VPS = (raw_score + Σw_neg) × 100                                    │
│                            ↓                                         │
├──────────────────────────────────────────────────────────────────────┤
│  LAYER 4: 5 KEY RISK FACTORS (Explanatory)                           │
│  [SA: Reputation] [SB: Malpractice] [SC: Medication]                 │
│  [SD: Outcomes] [SE: Burnout]                                        │
└──────────────────────────────────────────────────────────────────────┘
```

**Data Source:** MTS-Dialog Training Set (1,201 clinician-patient interactions)

---

## Repository Structure

```
├── Empathable_VPS_Framework_01.ipynb   # Core scoring framework & engine
├── Capstone_Working_Document.ipynb     # Main analysis pipeline
└── README.md
```

---

## Sub-Criteria Formulas

The 32 sub-criteria are the foundational inputs to the scoring engine. Each criterion is derived from a verified peer-reviewed study, with weights proportional to the reported effect size.

### Weight Configuration

| Category | Count | Raw Weight Sum | Normalized Weight |
|:---------|:-----:|:--------------:|:-----------------:|
| Positive sub-criteria | 28 | 2.720 | 85.27% |
| Negative sub-criteria | 4 | 0.470 | 14.73% |
| **Total** | **32** | **3.190** | **100%** |

### Normalization Formula

Each raw weight is normalized against the total:

```
normalized_weight[i] = raw_weight[i] / Σ(all raw weights)
```

### Positive Sub-Criteria (Selected)

| # | Sub-Criterion | Weight | Citation |
|:-:|:-------------|:------:|:---------|
| 1 | Empathy Expression | +15% | Howick et al. (2018) — SMD = −0.18 |
| 2 | Active Listening | +12% | Marvel et al. (1999) — RR ~2.3 |
| 4 | Teach-Back Method | +14.5% | Schillinger et al. (2003) — OR = 8.96 |
| 3 | Plain Language Use | +13.5% | AHRQ (2020) |
| 5 | Shared Decision Making | +12% | Barry & Edgman-Levitan (2012) |
| 32 | Positive Expectations | +12% | Howick et al. (2018) — SMD = −0.43 |
| 10 | Next Steps Clarity | +10.5% | Coleman et al. (2006) — 30% reduction |
| 22 | Red-Flag Education | +10% | Jack et al. (2009) — IRR = 0.70 |
| 7 | Rapport Building | +10% | Levinson et al. (1997) |
| 12 | Patient Autonomy Support | +9.5% | Barry & Edgman-Levitan (2012) |

### Negative Sub-Criteria

| # | Sub-Criterion | Penalty | Citation |
|:-:|:-------------|:-------:|:---------|
| 13 | Interruption | −13% | Marvel et al. (1999) — 72% interrupted at 23.1 sec |
| 15 | Rushing / Time Pressure | −11% | Levinson et al. (1997) — 3.3 min shorter visits |
| 16 | Unexplained Medical Jargon | −8% | Schillinger et al. (2003) — only 12% of concepts assessed |
| 14 | Dismissive Language | −13% | Hickson et al. (2002) — 5.8× complaint differential |

---

## VPS Scoring Framework

The **Value-to-Patient Score (VPS)** is calculated in two steps directly from sub-criteria scores. Skills and Risk Factors are *explanatory outputs*, not VPS inputs.

### Step 1 — Raw Score

```
raw_score = Σ(w_pos × score_pos) - Σ(w_neg × score_neg)
```

- `score_pos` and `score_neg` ∈ [0, 1] for each sub-criterion
- Theoretical raw range: [−0.1473, +0.8527]

### Step 2 — Rescale to [0, 100]

```
VPS = (raw_score + Σw_neg) × 100
    = (raw_score + 0.1473) × 100
```

This shift-and-scale guarantees:
- All positive = 1, all negative = 0 → **VPS = 100**
- All positive = 0, all negative = 1 → **VPS = 0**
- Average performance (all = 0.5) → **VPS = 50**

### Performance Bands

| Band | VPS Range | Color |
|:-----|:---------:|:-----:|
| Excellent | ≥ 85 | 🟢 Green |
| Good | 70 – 84 | 🟡 Yellow |
| Developing | 50 – 69 | 🟠 Orange |
| Needs Improvement | < 50 | 🔴 Red |

### 6 Core Skill Dimensions (Explanatory)

Skill scores explain *why* a clinician received a particular VPS. They are computed as weighted aggregates of the sub-criteria mapped to each skill.

| Skill | Key Sub-Criteria Inputs |
|:------|:------------------------|
| **Respect** | Empathy Expression, Active Listening, Cultural Sensitivity, Patient Autonomy |
| **Support** | Empathy Expression, Rapport Building, Positive Tone, Pain Acknowledgment |
| **Education** | Teach-Back, Plain Language, Red-Flag Education, Medication Communication |
| **Planning** | Next Steps Clarity, Shared Decision Making, Discharge Instructions, Care Transitions |
| **Engagement** | Active Listening, Shared Decision Making, Patient Autonomy, Responsiveness |
| **Communication** | Rapport Building, Plain Language, Individualized Explanation, Documentation Quality |

### 5 Risk Factors (Explanatory)

| Code | Risk | Key Drivers |
|:----:|:-----|:------------|
| SA | Reputation Damage | Low empathy, dismissive language, poor rapport |
| SB | Malpractice Risk | Interruption, informed consent quality, poor rapport |
| SC | Medication Errors | Jargon use, low teach-back, poor medication communication |
| SD | Poor Outcomes | Rushing, no next-steps clarity, no red-flag education |
| SE | Staff Burnout | Rushing, time pressure, low empathy expression |

---

## Resiliency Scoring Framework

The Resiliency Score measures a clinician's robustness across repeated interactions and adverse communication scenarios. It identifies patterns that persist under pressure — specifically, how consistently a clinician maintains effective communication behaviors when patient interactions are difficult, time-constrained, or emotionally complex.

### Components

**Behavioral Consistency Index (BCI):**
Tracks the standard deviation of sub-criteria scores across multiple interactions. Lower variance on high-weight criteria (e.g., Empathy Expression, Active Listening) signals a more resilient communicator.

```
BCI = 1 - σ(VPS scores across N interactions)
```

**Negative Behavior Suppression Rate (NBSR):**
Measures how infrequently the four penalized behaviors (Interruption, Dismissive Language, Rushing, Jargon) are detected across a clinician's interaction set.

```
NBSR = 1 - mean(score_neg across all interactions)
```

**Resiliency Score:**
```
Resiliency = α × BCI + β × NBSR
```
Where α and β are tunable weights (default: α = 0.6, β = 0.4).

### Interpretation

| Score Range | Label |
|:-----------:|:------|
| ≥ 0.85 | Highly Resilient |
| 0.70 – 0.84 | Resilient |
| 0.50 – 0.69 | Variable |
| < 0.50 | At Risk |

---

## Financial Model Framework

The Financial Model translates VPS and Risk Factor scores into dollar-denominated risk exposure and ROI estimates for healthcare institutions.

### Risk Quantification



---

## Baseline Creation

The baseline pipeline establishes a reference VPS for each clinician prior to any intervention, and validates the AI scoring engine against human raters.

### Human Scoring

A subset of interactions (N = 50) from the MTS-Dialog dataset were scored by trained human raters across all 6 RESPEC framework. Inter-rater reliability was assessed using Cohen's κ and intraclass correlation coefficient (ICC).

### Comparison to VPS Scores

Human-rated sub-criteria scores are passed through the `calculate_vps()` function to produce **Human VPS** scores. These are compared to the **Engine VPS** scores generated by the `EmpathableScoringEngine` (NLP-based inference) using:

- Pearson correlation (r) between Human VPS and Engine VPS
- Mean absolute error (MAE) per sub-criterion
- Bland-Altman agreement analysis

### Bayesian Update Model

After establishing human-AI score agreement, a Bayesian framework is used to produce calibrated, uncertainty-aware VPS estimates:

```
P(VPS | features) ∝ P(features | VPS) × P(VPS)
```

- **Prior:** Distribution of VPS scores from the full training set
- **Likelihood:** Scored sub-criteria features from the NLP engine
- **Posterior:** Updated VPS estimate with credible interval

This allows the model to express confidence in its score — particularly useful for edge-case interactions where linguistic signals are ambiguous. The posterior VPS replaces the point estimate in downstream reporting when uncertainty exceeds a defined threshold.

---

## LoRA: Patient Sub-Genre Modeling

Low-Rank Adaptation (LoRA) is used to fine-tune a base language model on patient sub-populations defined by communication archetype, enabling personalized scoring calibration.

### Motivation

The 32 sub-criteria weights are derived from population-level effect sizes. However, communication dynamics vary significantly across patient types — a patient with low health literacy requires fundamentally different clinician behaviors than a medically sophisticated patient presenting with a chronic condition. LoRA enables the scoring engine to adapt its evaluation lens to these sub-genres without full model retraining.

### Patient Sub-Genre Taxonomy

| Sub-Genre | Defining Characteristics | Key Sub-Criteria Emphasis |
|:----------|:------------------------|:--------------------------|
| Low Health Literacy | Simplified vocabulary, short responses, confusion signals | Plain Language, Teach-Back, Jargon penalty |
| High Anxiety / Emotional Distress | Affect-laden language, repetitive concerns, reassurance-seeking | Empathy Expression, Rapport Building, Support |
| Chronic Disease Management | Condition-specific vocabulary, adherence history, self-management language | Medication Communication, Next Steps, Care Transitions |
| Post-Discharge / Care Transition | Discharge-context vocabulary, follow-up uncertainty | Discharge Clarity, Red-Flag Education, Planning |
| Informed Consent Scenario | Procedure-specific language, decision-making signals | Informed Consent Quality, Shared Decision Making |

### LoRA Implementation

```python
# Conceptual structure
base_model = load_pretrained_model("clinical-bert-variant")

lora_config = LoRAConfig(
    r=8,                     # Low-rank dimension
    alpha=16,                # Scaling factor
    target_modules=["q", "v"],
    dropout=0.05
)

# Fine-tune one adapter per patient sub-genre
adapter = LoRAAdapter(base_model, lora_config)
adapter.train(sub_genre_dataset[sub_genre])
```

Each LoRA adapter produces sub-genre-adjusted sub-criteria scores. These are passed through the standard `calculate_vps()` pipeline, preserving full score interpretability.

---

## Visual Models

### Video



### Images



---

## Visualizations

### Alluvial  Diagram




## Research Foundation

All sub-criteria weights are grounded in peer-reviewed research. Key citations include:

- Howick et al. (2018) — *Journal of the Royal Society of Medicine*
- Schillinger et al. (2003) — *Archives of Internal Medicine*
- Marvel et al. (1999) — *JAMA*
- Levinson et al. (1997) — *JAMA*
- Hickson et al. (2002) — Provider communication & malpractice
- Barry & Edgman-Levitan (2012) — *NEJM*, shared decision making
- Coleman et al. (2006) — *Archives of Internal Medicine*, care transitions
- Jack et al. (2009) — *Annals of Internal Medicine*, Project RED
- AHRQ (2020) — Health Literacy Universal Precautions Toolkit

> **Verification status:** 130+ statistics confirmed against original publications. All verified citations are marked `VERIFIED` in `Empathable_VPS_Framework_01.ipynb`.

---

*Empathable Capstone Project — Built with the MTS-Dialog dataset*
