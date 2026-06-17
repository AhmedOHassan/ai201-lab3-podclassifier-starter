import json
import os
from config import VALID_LABELS, DATA_PATH, TEST_FILE
from classifier import classify_episode, load_labeled_examples


def run_evaluation() -> dict:
    """
    Run the classifier against the held-out test set and return full results.

    This function is already complete. It:
      1. Loads the labeled training examples (from your my_labels.json)
      2. Loads the test episodes (with ground-truth labels)
      3. Runs classify_episode() on each test description
      4. Returns a results dict with predictions, ground truth, and per-episode detail

    You'll use the results dict in compute_accuracy() and compute_per_class_accuracy().
    """
    labeled_examples = load_labeled_examples()

    test_path = os.path.join(DATA_PATH, TEST_FILE)
    with open(test_path, encoding="utf-8") as f:
        test_episodes = json.load(f)

    results = []
    for episode in test_episodes:
        print(f"  Classifying: {episode['title'][:60]}...")
        prediction = classify_episode(episode["description"], labeled_examples)
        results.append({
            "id": episode["id"],
            "title": episode["title"],
            "description": episode["description"],
            "ground_truth": episode["label"],
            "predicted": prediction["label"],
            "reasoning": prediction["reasoning"],
            "correct": prediction["label"] == episode["label"],
        })

    predictions = [r["predicted"] for r in results]
    ground_truth = [r["ground_truth"] for r in results]

    return {
        "results": results,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "total": len(results),
    }


def compute_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    """
    Compute overall classification accuracy.

    TODO — Milestone 3:

    Accuracy = number of correct predictions / total predictions.
    A prediction is correct when it exactly matches the ground truth label.

    Before writing code, complete specs/evaluation-spec.md.
    """
    # Basic validation
    if not predictions or not ground_truth:
        # If either list is empty, return 0.0 (no meaningful accuracy)
        return 0.0

    # Ensure lengths match; if not, only compare up to the shorter length
    n = min(len(predictions), len(ground_truth))
    if n == 0:
        return 0.0

    correct = 0
    for i in range(n):
        if predictions[i] == ground_truth[i]:
            correct += 1

    return correct / n


def compute_per_class_accuracy(
    predictions: list[str], ground_truth: list[str]
) -> dict[str, dict]:
    """
    Compute accuracy broken down by each label class.

    TODO — Milestone 3 (complete after compute_accuracy):

    For each label in VALID_LABELS, compute:
      - "correct"  : number of episodes with this ground-truth label predicted correctly
      - "total"    : number of episodes with this ground-truth label
      - "accuracy" : correct / total (0.0 if total is 0)

    Return a dict keyed by label. Example:
      {
        "interview": {"correct": 4, "total": 5, "accuracy": 0.8},
        "solo":      {"correct": 5, "total": 5, "accuracy": 1.0},
        ...
      }

    Before writing code, complete specs/evaluation-spec.md.
    """
    # Initialize counters
    stats = {label: {"correct": 0, "total": 0, "accuracy": 0.0} for label in VALID_LABELS}

    # Iterate over pairs
    n = min(len(predictions), len(ground_truth))
    for i in range(n):
        pred = predictions[i]
        truth = ground_truth[i]
        if truth in stats:
            stats[truth]["total"] += 1
            if pred == truth:
                stats[truth]["correct"] += 1

    # Compute accuracies, guard division by zero
    for label in VALID_LABELS:
        total = stats[label]["total"]
        correct = stats[label]["correct"]
        stats[label]["accuracy"] = (correct / total) if total > 0 else 0.0

    return stats


def format_evaluation_report(eval_results: dict) -> str:
    """
    Format evaluation results into a readable report string.

    This function is already complete. Pass it the dict returned by run_evaluation().
    """
    predictions = eval_results["predictions"]
    ground_truth = eval_results["ground_truth"]
    results = eval_results["results"]

    accuracy = compute_accuracy(predictions, ground_truth)
    per_class = compute_per_class_accuracy(predictions, ground_truth)

    lines = [
        f"## Evaluation Results\n",
        f"**Overall accuracy:** {accuracy:.1%} ({sum(r['correct'] for r in results)}/{eval_results['total']})\n",
        "\n**Per-class accuracy:**",
    ]
    for label, stats in per_class.items():
        bar = "█" * int(stats["accuracy"] * 10) + "░" * (10 - int(stats["accuracy"] * 10))
        lines.append(f"  {label:<12} {bar}  {stats['accuracy']:.0%}  ({stats['correct']}/{stats['total']})")

    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        lines.append(f"\n**Misclassified ({len(misclassified)}):**")
        for r in misclassified:
            lines.append(f"  [{r['ground_truth']} → {r['predicted']}] {r['title']}")
    else:
        lines.append("\n**No misclassifications — perfect score!**")

    return "\n".join(lines)
