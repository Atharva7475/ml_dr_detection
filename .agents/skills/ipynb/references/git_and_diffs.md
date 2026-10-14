/# Version Control, Diffs, and Git Cleanliness for Notebooks

Jupyter Notebooks (`.ipynb`) are notoriously challenging for Git because they store code, cell metadata, execution counts, and rich output blobs (such as base64-encoded PNGs) in a single JSON structure.

A single re-execution without any code changes can produce thousands of lines of noisy Git diffs and lead to nearly unresolvable merge conflicts.

---

## 1. Why Git Struggles with `.ipynb`

1. **Base64 Images**: A single `plt.show()` creates hundreds of kilobytes of alphanumeric base64 garbage in Git history.
2. **Execution Counts**: Running a cell changes `"execution_count": 1` to `"execution_count": 2`, triggering a diff on every run.
3. **Multiline String Arrays**: Small edits across cells create multi-line JSON diffs where conflict markers (`<<<<<<< HEAD`) break JSON syntax.

---

## 2. Solution 1: `nbstripout` (Automated Output Stripping)

`nbstripout` configures a Git filter that automatically strips cell outputs and execution counts before committing, while leaving your local working copy untouched.

### Installation
```bash
pip install nbstripout
```

### Git Integration (One-time Setup)
Run this inside your Git repository:
```bash
nbstripout install
```
This adds filter configurations to `.git/config` and `.gitattributes`:
```ini
# .gitattributes
*.ipynb filter=nbstripout
```

### Manual Stripping
To strip a notebook on demand:
```bash
# In-place stripping
nbstripout path/to/notebook.ipynb

# Verify if a notebook is clean
nbstripout --verify path/to/notebook.ipynb
```

---

## 3. Solution 2: `nbdime` (Content-Aware Diffing & Merging)

`nbdime` understands notebook semantics. It compares notebooks cell-by-cell rather than treating them as arbitrary JSON text.

### Installation
```bash
pip install nbdime
```

### Git Integration
Configure Git to use `nbdime` for all notebook diffs and merges:
```bash
nbdime config-git --enable --global
```
Now, running `git diff path/to/notebook.ipynb` shows clean, cell-level comparisons.

### Command-Line Usage
- **Terminal Diff**:
  ```bash
  nbdiff notebook_old.ipynb notebook_new.ipynb
  ```
- **Interactive Browser Diff (Side-by-Side Visual UI)**:
  ```bash
  nbdiff-web notebook_old.ipynb notebook_new.ipynb
  ```
- **3-Way Semantic Merge**:
  ```bash
  nbmerge base.ipynb local.ipynb remote.ipynb --out merged.ipynb
  ```

---

## 4. Solution 3: Jupytext (Plain-Text Paired Files)

Jupytext pairs an `.ipynb` file with a plain-text Python script (`.py`) formatted with `# %%` cell markers (the "percent" format).

### Why Paired Formats?
- Standard IDEs (VS Code, Cursor, PyCharm) treat the `.py` file as regular Python with full LSP, autocomplete, and refactoring support.
- Git tracks the clean `.py` file with 100% human-readable diffs.
- Jupyter opens and executes the notebook with interactive cell visualization.

### Setting up Paired Notebooks via CLI
```bash
# Pair an existing notebook with a percent format Python script
jupytext --set-formats ipynb,py:percent notebook.ipynb

# Sync changes from .py back to .ipynb
jupytext --sync notebook.ipynb

# Convert on demand
jupytext --to py:percent notebook.ipynb
jupytext --to notebook script.py
```

---

## 5. Built-in Agent Utility: `ipynb_tool`

If external tools like `nbstripout` or `nbdime` are not installed in the environment, use the bundled `ipynb_tool.py`:

```bash
# Clear outputs from all cells (identical to nbstripout)
python .agents/skills/ipynb/scripts/ipynb_tool.py clear-outputs notebook.ipynb

# Compare two notebooks cell-by-cell in the terminal
python .agents/skills/ipynb/scripts/ipynb_tool.py diff nb_v1.ipynb nb_v2.ipynb

# Export to percent-format Python script
python .agents/skills/ipynb/scripts/ipynb_tool.py to-py notebook.ipynb --output script.py

# Import from percent-format Python script
python .agents/skills/ipynb/scripts/ipynb_tool.py from-py script.py --output notebook.ipynb
```
