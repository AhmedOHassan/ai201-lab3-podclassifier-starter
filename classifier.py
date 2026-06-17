import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    # Prepare balanced, limited examples for few-shot (up to 8 examples, max 2 per label)
    examples_by_label: dict = {label: [] for label in VALID_LABELS}
    for ex in labeled_examples:
        lab = ex.get("label")
        if lab in examples_by_label:
            examples_by_label[lab].append(ex)

    selected = []
    per_label = 2
    for lab in VALID_LABELS:
        items = examples_by_label.get(lab, [])[:per_label]
        selected.extend(items)

    # Build examples text
    example_blocks = []
    for ex in selected:
        block = f"Title: {ex.get('title','')}\nDescription: {ex.get('description','')}\nLabel: {ex.get('label')}"
        example_blocks.append(block)

    examples_text = "\n\n---\n\n".join(example_blocks)

    instruction = (
        "You are classifying podcast episodes by their format. "
        "Assign exactly one of these labels: " + ", ".join(VALID_LABELS) + ".\n\n"
        "Definitions:\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — no guests, no assembled external sources\n"
        "- panel: three or more speakers discussing a topic together with roughly equal standing\n"
        "- narrative: a story assembled from external sources (reporting, archives, interviews) with a clear arc\n\n"
    )

    prompt_parts = [instruction]
    if examples_text:
        prompt_parts.append("Here are some labeled examples (demonstrations):\n\n" + examples_text)
    else:
        prompt_parts.append("No labeled examples are available. Classify using the definitions above.")

    # The episode to classify
    prompt_parts.append(
        f"\n\nNow classify the following episode in the same format as the examples:\nTitle: {''}\nDescription: {description}\nLabel: ?\n\n"
    )

    # Output format: request strict JSON only
    prompt_parts.append(
        "Return ONLY a JSON object (no surrounding text) with exactly two keys:\n"
        "{\n  \"label\": <one of: " + ", ".join(VALID_LABELS) + ">,\n  \"reasoning\": <a brief justification string>\n}\n"
        "Example valid output: {\"label\": \"interview\", \"reasoning\": \"Host asks an expert about their research.\"}\n"
    )

    return "\n".join(prompt_parts)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """
    prompt = build_few_shot_prompt(labeled_examples, description)

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        raw = response.choices[0].message.content
        # Helpful for debugging while developing parsing logic
        print("LLM raw response:\n", raw)

        # Try to extract a JSON object from the response
        label = None
        reasoning = None

        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw[start : end + 1]
                parsed = json.loads(candidate)
                label = parsed.get("label")
                reasoning = parsed.get("reasoning", "")
        except Exception:
            label = None
            reasoning = None

        # Fallback parsing (line-based)
        if not label:
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                lower = line.lower()
                if lower.startswith("label:"):
                    label = line.split(":", 1)[1].strip().strip('"')
                elif lower.startswith("reasoning:") or lower.startswith("explanation:"):
                    reasoning = line.split(":", 1)[1].strip().strip('"')

        # If we still have no reasoning but have raw, set it to raw
        if not reasoning:
            reasoning = raw.strip()

        # Validate label
        if not isinstance(label, str) or label not in VALID_LABELS:
            final_label = "unknown"
        else:
            final_label = label

        return {"label": final_label, "reasoning": reasoning}

    except Exception as e:
        # On any error, return unknown and include a short error message
        snippet = ""
        try:
            snippet = raw[:200]
        except Exception:
            snippet = "(no raw response)"
        return {"label": "unknown", "reasoning": f"error: {e}; raw: {snippet}"}
