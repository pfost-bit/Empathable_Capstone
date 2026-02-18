"""
VPS Validation - Module 1: Annotation Interface
================================================
A terminal-based tool for human raters to label conversation snippets
across the 6 RESPEC dimensions. Saves results to CSV for later analysis.

Usage:
    python 1_annotation_interface.py --rater "Alice" --input conversations.csv
    python 1_annotation_interface.py --rater "Bob"   --input conversations.csv
"""

import argparse
import csv
import json
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

# ── Synthetic example conversations (replace with your real data) ─────────────

EXAMPLE_CONVERSATIONS = [
    {
        "id": "conv_001",
        "text": (
            "Doctor: So what brings you in today?\n"
            "Patient: I've been having these headaches almost every day for two weeks.\n"
            "Doctor: Okay. How long do they last?\n"
            "Patient: A few hours usually. It's really affecting my work.\n"
            "Doctor: We'll do some tests. Come back in a week."
        ),
    },
    {
        "id": "conv_002",
        "text": (
            "Doctor: I can see you've been dealing with these headaches for a while — that sounds exhausting.\n"
            "Patient: It really is. I'm struggling to get through the day.\n"
            "Doctor: I completely understand. Let's figure this out together. Can you tell me more "
            "about when they tend to happen — morning, evening, after certain activities?\n"
            "Patient: Mostly in the afternoon, after staring at screens.\n"
            "Doctor: That's a helpful clue. There are a few likely causes I want to walk you through, "
            "and then we'll decide together what makes sense to try first."
        ),
    },
    {
        "id": "conv_003",
        "text": (
            "Patient: I'm really worried about these chest pains.\n"
            "Doctor: When did they start?\n"
            "Patient: About three days ago. Could it be my heart?\n"
            "Doctor: Probably not. You're young. Take an antacid and see if it helps.\n"
            "Patient: Should I be monitoring anything?\n"
            "Doctor: Just come back if it gets worse."
        ),
    },
]

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
    """Load from CSV with columns: id, text. Falls back to built-in examples."""
    if not path or not os.path.exists(path):
        print(f"\n  ℹ  File '{path}' not found — using built-in example conversations.\n")
        return EXAMPLE_CONVERSATIONS

    convs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            convs.append({"id": row["id"], "text": row["text"]})
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
    print(f"\n  You will score {total} conversation(s) across 6 RESPEC skills.")
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
    parser.add_argument("--input",  default=None,  help="Path to conversations CSV (id, text columns)")
    parser.add_argument("--output", default="annotations", help="Output directory for annotation CSVs")
    args = parser.parse_args()

    conversations = load_conversations(args.input)
    annotate(rater=args.rater, conversations=conversations, output_dir=args.output)


if __name__ == "__main__":
    main()
