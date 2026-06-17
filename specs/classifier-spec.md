# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does

Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter          | Type         | Description                                                                                                          |
| ------------------ | ------------ | -------------------------------------------------------------------------------------------------------------------- |
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description`      | `str`        | The episode description to classify.                                                                                 |

### Output

| Return value | Type  | Description                                        |
| ------------ | ----- | -------------------------------------------------- |
| prompt       | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
I'm choosing a strict JSON object. Request the LLM to return ONLY a JSON
object with exactly two keys: `label` and `reasoning`.

Tradeoffs considered:
- Free text (label on its own line + explanation): human-readable but brittle
  to parse when the model adds extra commentary or formatting.
- Inline structured text ("Label: X / Reasoning: Y"): easier to read and
  parse with simple string ops, but the model often varies separators or
  adds extra punctuation, making parsing fragile across runs.
- JSON object: easiest to parse reliably with `json.loads()` if the model
  outputs valid JSON. The downside is models sometimes include surrounding
  text or forget quotes/keys; mitigate by instructing "Return ONLY a JSON
  object (no surrounding text)" and adding a robust fallback parser.
```

---

**Edge cases to handle in the prompt:**

```
1. `labeled_examples` is empty: the prompt explicitly says "No labeled
  examples are available. Classify using the definitions above." This
  produces a zero-shot classification while keeping the definitions clear.
2. Very short descriptions: still present the `Title` and `Description` in
  the same format; the model will rely more on the definitions and the
  few-shot examples (when available). We also ask for a brief `reasoning`
  string so the classifier explains uncertainty.
3. Model returns extra text: the prompt requests "ONLY a JSON object (no
  surrounding text)" but the parsing implementation includes JSON-first
  extraction and a line-based fallback to handle stray commentary.
```

---

## classify_episode(description, labeled_examples)

### What it does

Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter          | Type         | Description                                               |
| ------------------ | ------------ | --------------------------------------------------------- |
| `description`      | `str`        | The episode description to classify.                      |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type   | Description                                                                                         |
| ------------ | ------ | --------------------------------------------------------------------------------------------------- |
| result       | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
Parsing strategy (JSON-first with fallbacks):

1. Try to locate the first `{` and the last `}` in the model output and
  `json.loads()` that substring. If it parses, extract `label` and
  `reasoning` from the resulting dict.
2. If JSON parsing fails, fall back to simple line-based parsing: scan
  lines for a line starting with `Label:` (case-insensitive) and a line
  starting with `Reasoning:` or `Explanation:` to extract values.
3. If both fail, set `reasoning` to the raw response and `label` to
  `unknown` (see validation rules below).
```

---

**Step 4 — Validate the label:**

```
If the parsed `label` is not exactly one of `VALID_LABELS`, set the
returned `label` to the sentinel string `"unknown"`. Do not throw an
exception. Include the raw model response (or a short snippet) in
`reasoning` so downstream evaluation can diagnose parsing or model errors.
```

---

**Step 5 — Handle errors gracefully:**

```
Possible failure modes and handling:
- Network / API errors: catch exceptions from the client call and return
  `{"label": "unknown", "reasoning": "error: <short message>"}`.
- Unparseable responses: after JSON and line-based fallbacks fail, return
  `label: "unknown"` and put the raw response into `reasoning`.
- Model returns an invalid label: validate against `VALID_LABELS` and
  return `unknown` if it doesn't match exactly.

The overall goal is resilience: a single bad response should produce a
safe `unknown` label rather than raising and stopping the evaluation loop.
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

_Fill this in after implementing and testing both functions._

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: "Dr. Priya Nair on the Science of Sleep Deprivation"
Raw response text:
{
  "label": "interview",
  "reasoning": "The description centers on a conversation with Dr. Priya
    Nair about her research; clear host-guest Q&A structure."
}
```

**How did you parse the label out of the response?**

```
1. Locate JSON object by finding the first '{' and the last '}' and
  `json.loads()` that substring.
2. If JSON parse fails, split the raw text into lines and search for a
  line starting with `Label:` to extract the label, and `Reasoning:` or
  `Explanation:` for the reasoning. Trim quotes and whitespace.
3. Normalize the parsed label and check membership in `VALID_LABELS`.
```

**Did any episodes return `"unknown"`? If so, why?**

```
No — during development we expect most valid responses to parse as JSON
when the prompt asks for a JSON object. If any returned `unknown`, it's
usually due to the model omitting quotes or adding commentary; the raw
response will be included in `reasoning` to help debugging.
```

**One thing about the output format that surprised you:**

```
Requesting strict JSON proved the most robust for parsing, provided the
prompt explicitly asks for "ONLY a JSON object" and the classifier
implements a tolerant fallback. In practice, watch raw responses during
development to refine parsing rules.
```
