"""
VPS Validation - Module 1: Annotation Interface
================================================
A terminal-based tool for human raters to label conversation snippets
across the 6 RESPEC dimensions. Saves results to CSV for later analysis.

Usage:
    python 1_annotation_interface.py --rater "Alice" --input conversations.csv
    python 1_annotation_interface.py --rater "Bob"   --input conversations.csv

Input CSV must have exactly 50 rows with columns: id, text.
"""

import argparse
import csv
import os
import textwrap
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

SKILLS = ["Respect", "Support", "Education", "Planning", "Engagement", "Communication"]

SKILL_DESCRIPTIONS = {
    "Respect":       "Does the doctor acknowledge the patient's dignity, autonomy, and feelings?",
    "Support":       "Does the doctor offer emotional support or validation?",
    "Education":     "Does the doctor clearly explain information or answer questions?",
    "Planning":      "Does the doctor collaboratively involve the patient in next steps?",
    "Engagement":    "Does the doctor actively listen and encourage the patient to share?",
    "Communication": "Is the doctor's language clear, warm, and appropriate?",
}

SCALE = {1: "Very Poor", 2: "Poor", 3: "Neutral/Average", 4: "Good", 5: "Excellent"}

REQUIRED_ROW_COUNT = 50

# ── Helpers ───────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def wrap(text: str, width: int = 58) -> str:
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width) or [""])
    return "\n".join(wrapped)


def get_score(skill: str) -> int:
    """Prompt rater for a 1-5 score for a given skill."""
    desc = SKILL_DESCRIPTIONS[skill]
    scale_str = "  ".join(f"{k}={v}" for k, v in SCALE.items())
    print(f"\n  [{skill}]  {desc}")
    print(f"  Scale: {scale_str}")
    while True:
        raw = input("  Your score (1-5): ").strip()
        if raw.isdigit() and int(raw) in SCALE:
            return int(raw)
        print("  ⚠  Please enter a number from 1 to 5.")


def get_notes() -> str:
    return input("\n  Optional notes (press Enter to skip): ").strip()


def load_conversations(path: str) -> list[dict]:
    """Load exactly 50 conversations from a CSV with columns: id, text."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: '{path}'\n"
            "Please provide a valid CSV with columns: id, text."
        )

    convs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [col for col in ("id", "text") if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"CSV is missing required column(s): {missing}\n"
                "Expected columns: id, text."
            )
        for row in reader:
            convs.append({"id": row["id"], "text": row["text"]})

    if len(convs) != REQUIRED_ROW_COUNT:
        raise ValueError(
            f"Expected exactly {REQUIRED_ROW_COUNT} conversations, "
            f"but found {len(convs)} in '{path}'."
        )

    return convs


def save_annotations(annotations: list[dict], rater: str, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"annotations_{rater}_{timestamp}.csv")

    fieldnames = ["conv_id", "rater", "timestamp"] + SKILLS + ["notes"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotations)

    print(f"\n  ✅  Saved {len(annotations)} annotations → {filename}")
    return filename


# ── Main annotation loop ──────────────────────────────────────────────────────

def annotate(rater: str, conversations: list[dict], output_dir: str = "annotations"):
    annotations = []
    total = len(conversations)

    print_header(f"VPS Annotation Tool  |  Rater: {rater}")
    print(f"\n  You will score {total} conversations across 6 RESPEC skills.")
    print("  Use scale 1 (Very Poor) → 5 (Excellent).")
    input("\n  Press Enter to begin...\n")

    for i, conv in enumerate(conversations, 1):
        clear()
        print_header(f"Conversation {i} of {total}  |  ID: {conv['id']}")
        print()
        print(wrap(conv["text"]))
        print()

        scores = {}
        for skill in SKILLS:
            scores[skill] = get_score(skill)

        notes = get_notes()

        row = {
            "conv_id":   conv["id"],
            "rater":     rater,
            "timestamp": datetime.now().isoformat(),
            "notes":     notes,
            **scores,
        }
        annotations.append(row)

        print(f"\n  ✔  Recorded scores for conversation {i}.")
        if i < total:
            input("  Press Enter for next conversation...")

    clear()
    print_header("Annotation Complete")
    return save_annotations(annotations, rater, output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VPS Human Annotation Interface")
    parser.add_argument("--rater",  required=True, help="Rater name or ID (e.g. 'Alice')")
    parser.add_argument("--input",  required=True, help="Path to conversations CSV (id, text columns) — must have exactly 50 rows")
    parser.add_argument("--output", default="annotations", help="Output directory for annotation CSVs")
    args = parser.parse_args()

    conversations = load_conversations(args.input)
    annotate(rater=args.rater, conversations=conversations, output_dir=args.output)


if __name__ == "__main__":
    main()
