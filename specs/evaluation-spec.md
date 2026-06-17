# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:

- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does

Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter      | Type        | Description                                                |
| -------------- | ----------- | ---------------------------------------------------------- |
| `predictions`  | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`.    |

### Output

| Return value | Type    | Description                  |
| ------------ | ------- | ---------------------------- |
| accuracy     | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
Accuracy = (number of predictions that exactly match the corresponding
ground-truth labels) divided by (the total number of predictions).

"Correct" means the predicted label string is exactly equal to the
ground-truth label string at the same index.
```

---

**Step-by-step logic:**

```
1. Verify `predictions` and `ground_truth` have the same length.
2. Iterate over paired elements (pred, truth) and count matches where
   `pred == truth`.
3. Compute accuracy = matches / len(predictions). Return a float.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. With zero examples there is no meaningful accuracy; returning
0.0 avoids division-by-zero and signals that no correct predictions were
made. (Alternatives include returning `None`, but the evaluation pipeline
expects a numeric value.)
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Compare index-by-index:
0: interview == interview → correct
1: solo == solo → correct
2: panel != solo → incorrect
3: interview != narrative → incorrect

Matches = 2, Total = 4 → accuracy = 2 / 4 = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does

Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter      | Type        | Description                               |
| -------------- | ----------- | ----------------------------------------- |
| `predictions`  | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order.        |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
An episode counts as correctly classified for class `C` when its ground
truth label equals `C` and the prediction for that episode also equals `C`.
For example, an episode with ground-truth `interview` is "correct" if the
prediction is exactly `interview`.
```

---

**What does "total" mean for a given class?**

```
`total` is the number of test episodes whose ground-truth label equals the
class. It is not the number of predictions equal to the class; it is the
number of opportunities to correctly predict that class.
```

---

**Step-by-step logic:**

```
1. Initialize a dict mapping each label in `VALID_LABELS` to counters
   `correct=0` and `total=0`.
2. Iterate over paired lists (predicted, truth).
3. For each pair, increment `total` for the `truth` label by 1. If
   `predicted == truth`, also increment `correct` for that label.
4. After the loop, compute `accuracy = correct / total` for each label.
   If `total == 0`, set `accuracy = 0.0`.
5. Return the per-label dict with keys `correct`, `total`, and `accuracy`.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set `accuracy` to 0.0 when `total == 0`. This avoids division-by-zero and
keeps the return type uniform (float). The docstring in `evaluate.py`
documents this behavior.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

Compute totals and corrects by scanning each index:
0: pred=interview, truth=interview → interview correct
1: pred=interview, truth=solo → solo total +1, not correct
2: pred=solo, truth=solo → solo correct
3: pred=panel, truth=panel → panel correct
4: pred=panel, truth=narrative → narrative total +1, not correct

Results:
label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0
solo        1        2      0.5
panel       1        1      1.0
narrative   0        1      0.0

label       correct  total  accuracy
----------  -------  -----  --------
interview   [blank]  [blank]  [blank]
solo        [blank]  [blank]  [blank]
panel       [blank]  [blank]  [blank]
narrative   [blank]  [blank]  [blank]
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
