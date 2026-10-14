# Workflow Example: Safely Inspecting and Editing a Notebook

This walkthrough demonstrates the end-to-end workflow an agent or developer follows to inspect and modify code cells in an existing notebook without risking JSON corruption or exceeding token limits.

---

## Scenario
The user asks:
> *"In `notebooks/04_efficientnet_cropped384_experiment.ipynb`, change the batch size from 16 to 32 and add a CosineAnnealingLR scheduler."*

---

## Step 1: Generate an Outline
Instead of reading the entire multi-megabyte `.ipynb` file, generate a concise outline:

```bash
python .agents/skills/ipynb/scripts/ipynb_tool.py outline notebooks/04_efficientnet_cropped384_experiment.ipynb
```

**Example Output:**
```text
=== Notebook: 04_efficientnet_cropped384_experiment.ipynb (9 cells) ===
[  0 | md   |      |  2L]: # EfficientNet Training with Cropped 384x384 Inputs
[  1 | code | #1   |  8L [outputs: 1]]: import os; import torch
[  2 | code | #2   | 14L [outputs: 0]]: BATCH_SIZE = 16\nLEARNING_RATE = 1e-4
[  3 | code | #3   | 28L [outputs: 1]]: def build_dataloader(): ...
[  4 | code | #4   | 45L [outputs: 0]]: def train_one_epoch(): ...
...
```
*Insight:* Cell `2` contains the hyperparameters (`BATCH_SIZE`), and Cell `4` contains the training loop.

---

## Step 2: Read Only the Target Cell
Read Cell 2 with full code visibility:

```bash
python .agents/skills/ipynb/scripts/ipynb_tool.py read notebooks/04_efficientnet_cropped384_experiment.ipynb --cell 2
```

**Output:**
```text
--- [CELL 2 | CODE | exec:2] ---
```python
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 15
WEIGHT_DECAY = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```
```

---

## Step 3: Perform the Surgical Edit
Apply the modification with the `--diff` flag to verify the change:

```bash
python .agents/skills/ipynb/scripts/ipynb_tool.py edit notebooks/04_efficientnet_cropped384_experiment.ipynb \
  --cell 2 \
  --content "BATCH_SIZE = 32
LEARNING_RATE = 2e-4
EPOCHS = 15
WEIGHT_DECAY = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'" \
  --diff
```

**Output:**
```text
Successfully updated Cell 2 in 'notebooks/04_efficientnet_cropped384_experiment.ipynb'.
Diff of changes:
--- cell_2_before
+++ cell_2_after
@@ -1,5 +1,5 @@
-BATCH_SIZE = 16
-LEARNING_RATE = 1e-4
+BATCH_SIZE = 32
+LEARNING_RATE = 2e-4
 EPOCHS = 15
 WEIGHT_DECAY = 1e-2
 DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Step 4: Validate the Notebook Structure
Verify that the notebook conforms to the official nbformat v4 schema:

```bash
python .agents/skills/ipynb/scripts/ipynb_tool.py validate notebooks/04_efficientnet_cropped384_experiment.ipynb
```

**Output:**
```text
VALID: 'notebooks/04_efficientnet_cropped384_experiment.ipynb' conforms to nbformat v4 schema (9 cells).
```

---

## Step 5: (Optional) Sanitize Outputs Before Git Commit
If outputs are no longer relevant or need to be stripped for a clean git diff:

```bash
python .agents/skills/ipynb/scripts/ipynb_tool.py clear-outputs notebooks/04_efficientnet_cropped384_experiment.ipynb
```
