# Agent Troubleshooting & Safe Operation Guide

This guide is specifically designed for AI agents working with Jupyter Notebooks (`.ipynb`). It addresses agent-specific tool constraints, context window protection, and robust error recovery patterns.

---

## 1. The Core Tool Constraint

In Antigravity and standard LLM toolchains:
> **`replace_file_content` and `multi_replace_file_content` explicitly prohibit editing `.ipynb` files.**

Attempting to run `replace_file_content` on an `.ipynb` file will trigger an immediate tool error.

### Why this rule exists:
1. `.ipynb` files are strict JSON. Attempting string replacement on JSON blobs frequently leaves trailing commas, unescaped newlines (`\n`), broken brackets, or corrupts schema validation.
2. Even minor formatting changes can inadvertently strip mandatory notebook metadata keys (`nbformat`, `nbformat_minor`, `cell_id`).

### The Recommended Solution:
**Always use `ipynb_tool.py` or `python -c "import nbformat..."` scripts to read and manipulate notebooks.**

---

## 2. Preventing Context Window Exhaustion

Large notebooks often contain:
- Megabytes of base64-encoded PNG/JPEG plots (`data:image/png;base64,iVBOR...`)
- Thousands of lines of stdout from loops or epochs (`loss: 0.432, step 10/1000...`)

### Bad Pattern:
Calling `view_file` on a 5MB `.ipynb` file dumps raw JSON into the agent's context, wasting thousands of tokens and hitting output truncation boundaries.

### Best Practice:
1. **First, inspect the outline:**
   ```bash
   python .agents/skills/ipynb/scripts/ipynb_tool.py outline path/to/notebook.ipynb
   ```
   *Output:* A compact 10-20 line summary showing cell indices, types, line counts, and first lines.
2. **Second, inspect only target cells:**
   ```bash
   python .agents/skills/ipynb/scripts/ipynb_tool.py read path/to/notebook.ipynb --cell 3
   ```
   *Output:* Clean Markdown and Python code blocks. Outputs are sanitized and truncated; base64 images are collapsed into harmless placeholders like `<image/png attachment, ~45 KB>`.

---

## 3. Surgical Cell Edits

### Inline Edits (Short Changes)
```bash
python .agents/skills/ipynb/scripts/ipynb_tool.py edit path/to/notebook.ipynb \
    --cell 3 \
    --content "learning_rate = 5e-4\nbatch_size = 32" \
    --diff
```

### Complex Multi-Line Edits (via Patch File)
For large functions or classes:
1. Write the new Python code to a temporary file (e.g., in `scratch/new_train_loop.py`).
2. Apply the edit:
   ```bash
   python .agents/skills/ipynb/scripts/ipynb_tool.py edit path/to/notebook.ipynb \
       --cell 5 \
       --file scratch/new_train_loop.py \
       --diff
   ```
3. Remove the temporary patch file.

---

## 4. Recovering from Corrupted Notebooks

If a notebook fails to open in Jupyter or throws `JSONDecodeError` / `ValidationError`:

### Step 1: Diagnose
```bash
python .agents/skills/ipynb/scripts/ipynb_tool.py validate path/to/notebook.ipynb
```

### Step 2: Auto-repair Missing Cell IDs or Metadata
In nbformat v4.5+, each cell requires a unique `id` string:
```python
import nbformat
import uuid

with open("path/to/notebook.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if "id" not in cell or not cell["id"]:
        cell["id"] = str(uuid.uuid4())[:8]

nbformat.validate(nb)

with open("path/to/notebook.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
```

---

## 5. Handling Headless Execution Timeouts & Hangs

If running `ipynb_tool execute` hangs:
1. Ensure `--timeout 300` is passed so stuck cells auto-terminate after 5 minutes.
2. Check if GUI plot windows are blocking execution. Prepend this cell at index 0:
   ```python
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   plt.ioff()
   ```
3. Check working directory (`--cwd`): Notebooks that use relative filepaths like `open('data.csv')` will fail if executed from a different directory. Pass `--cwd <folder>`.
