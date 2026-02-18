# VPS Sentiment Analysis Validation Toolkit

Three-module pipeline for validating your NLP model against human-annotated ground truth.

## Workflow

```
1_annotation_interface.py   → collect human labels
        ↓
2_inter_rater_reliability.py → confirm raters agree (κ, α)
        ↓
3_correlation_analysis.py   → compare model vs. human (ρ, MAE, bias)
```

---

## Module 1 — Annotation Interface

Terminal-based tool for raters to score conversations across all 6 RESPEC skills.

```bash
# Each rater runs this independently
python 1_annotation_interface.py --rater "Alice" --input conversations.csv
python 1_annotation_interface.py --rater "Bob"   --input conversations.csv
```

**Input CSV format:**
```
id,text
conv_001,"Doctor: ... Patient: ..."
conv_002,"Doctor: ..."
```

**Output:** `annotations/annotations_Alice_<timestamp>.csv`

---

## Module 2 — Inter-Rater Reliability

Run *before* using human labels as ground truth. Confirms raters agree enough.

```bash
python 2_inter_rater_reliability.py --dir annotations/

# Or specify two files explicitly
python 2_inter_rater_reliability.py \
    --r1 annotations/annotations_Alice_20250101.csv \
    --r2 annotations/annotations_Bob_20250101.csv
```

**Key metrics:**
- **Krippendorff's α** — primary metric, works for 2+ raters on ordinal scale
- **Cohen's κ (linear-weighted)** — pairwise, penalizes larger disagreements
- **Spearman ρ** — rank correlation between raters

**Thresholds:**
| α / κ      | Interpretation                  |
|------------|----------------------------------|
| ≥ 0.80     | Substantial — very trustworthy  |
| 0.60–0.79  | Moderate — acceptable           |
| 0.40–0.59  | Fair — calibrate before using   |
| < 0.40     | Poor — raters need alignment    |

---

## Module 3 — Correlation Analysis

Compares your NLP model's scores to the human consensus.

```bash
python 3_correlation_analysis.py \
    --model model_scores.csv \
    --human annotations/ \
    --output results/

# Demo mode (no data needed)
python 3_correlation_analysis.py --demo
```

**Model scores CSV format:**
```
conv_id,Respect,Support,Education,Planning,Engagement,Communication
conv_001,3.2,4.1,2.8,3.5,3.0,3.7
```

**Outputs:**
- `results/correlation_results.csv` — Spearman ρ, MAE, bias per skill
- `results/scatter_grid.png` — model vs. human scatter per skill
- `results/bland_altman.png` — agreement + bias visualization
- `results/correlation_summary.png` — summary bar charts

---

## Dependencies

```bash
pip install pandas numpy scipy matplotlib scikit-learn
```

---

## Connecting to Your Existing Scoring Code

In `3_correlation_analysis.py`, your model scores should be pre-generated and saved as a CSV. To generate them from your existing VPS script:

```python
# In your capstone script, after computing scores:
results = []
for conv_id, conversation_text in your_conversations.items():
    scores = your_vps_scoring_function(conversation_text)
    scores["conv_id"] = conv_id
    results.append(scores)

pd.DataFrame(results).to_csv("model_scores.csv", index=False)
```

Then pass `model_scores.csv` to `3_correlation_analysis.py`.
