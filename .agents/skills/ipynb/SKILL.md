---
name: ipynb
description: >-
  Comprehensive guide and toolset for working with Jupyter Notebooks (.ipynb).
  Use this skill whenever inspecting, reading, editing, creating, running,
  clearing outputs, diffing, converting, or testing .ipynb notebook files.
---

# Jupyter Notebook (`.ipynb`) Skill

A production-grade skill and toolset for safely manipulating, executing, version-controlling, and inspecting Jupyter Notebooks (`.ipynb`).

---

## ⚠️ Critical Rule for AI Agents

> **NEVER attempt to edit `.ipynb` files using `replace_file_content` or `multi_replace_file_content`.**
> The system tool definitions explicitly prohibit modifying `.ipynb` files directly via text replacement tools. Furthermore, manual text manipulation of raw JSON breaks notebook schemas, corrupts execution counters, and risks syntax errors.
>
> **ALWAYS use the bundled helper script [`ipynb_tool.py`](./scripts/ipynb_tool.py) or `nbformat` to inspect and modify notebooks.**

---

## Quick Reference: `ipynb_tool.py` CLI

The skill provides an all-in-one CLI utility located at:
`python .agents/skills/ipynb/scripts/ipynb_tool.py <subcommand>`

| Subcommand | Purpose | Key Arguments | Example |
| :--- | :--- | :--- | :--- |
| **`outline`** | Token-efficient summary of cell types & contents | `notebook` | `python .../ipynb_tool.py outline exp.ipynb` |
| **`read`** | View cell code & clean outputs (collapses base64) | `--cell`, `--start`, `--end`, `--no-outputs` | `python .../ipynb_tool.py read exp.ipynb --cell 3` |
| **`edit`** | Replace cell source code safely | `--cell`, `--content`, `--file`, `--diff` | `python .../ipynb_tool.py edit exp.ipynb --cell 3 --content "lr=1e-4" --diff` |
| **`insert`** | Insert new code / markdown cell | `--index`, `--type`, `--content`, `--tags` | `python .../ipynb_tool.py insert exp.ipynb --index 1 --content "# Config"` |
| **`delete`** | Delete cell at index | `--cell` | `python .../ipynb_tool.py delete exp.ipynb --cell 4` |
| **`clear-outputs`** | Strip execution counts & outputs (clean git) | `--output` | `python .../ipynb_tool.py clear-outputs exp.ipynb` |
| **`execute`** | Headless run via `nbclient` | `--timeout`, `--cwd`, `--output` | `python .../ipynb_tool.py execute exp.ipynb --timeout 300` |
| **`to-py`** | Convert to `# %%` percent Python script | `--output` | `python .../ipynb_tool.py to-py exp.ipynb --output exp.py` |
| **`from-py`** | Convert `# %%` percent Python script to `.ipynb` | `python_file`, `--output` | `python .../ipynb_tool.py from-py exp.py --output exp.ipynb` |
| **`diff`** | Cell-by-cell semantic diff between 2 notebooks | `nb1`, `nb2` | `python .../ipynb_tool.py diff old.ipynb new.ipynb` |
| **`validate`** | Validate notebook adheres to nbformat v4 schema | `notebook` | `python .../ipynb_tool.py validate exp.ipynb` |

---

## Core Agent Procedures

### Procedure 1: Inspecting an Unfamiliar Notebook
Avoid calling `view_file` directly on an `.ipynb` file. Base64-encoded plot images and verbose outputs easily exceed token limits.

1. **Get the Outline First**:
   ```bash
   python .agents/skills/ipynb/scripts/ipynb_tool.py outline path/to/notebook.ipynb
   ```
2. **Read Specific Cells of Interest**:
   ```bash
   python .agents/skills/ipynb/scripts/ipynb_tool.py read path/to/notebook.ipynb --cell <index>
   ```
   *Note: Text outputs are truncated cleanly, and base64 plots are summarized as `<image/png attachment, ~45 KB>`.*

---

### Procedure 2: Surgically Editing a Cell
To update code in an existing notebook without breaking JSON structure:

1. Identify the target cell index from the outline.
2. For small changes, pass `--content`:
   ```bash
   python .agents/skills/ipynb/scripts/ipynb_tool.py edit path/to/notebook.ipynb \
     --cell <index> \
     --content "import torch\nDEVICE = 'cuda'" \
     --diff
   ```
3. For larger multi-line code blocks, write the replacement code to a temporary file in `scratch/patch.py` and pass `--file`:
   ```bash
   python .agents/skills/ipynb/scripts/ipynb_tool.py edit path/to/notebook.ipynb \
     --cell <index> \
     --file scratch/patch.py \
     --diff
   ```
4. Verify schema validity:
   ```bash
   python .agents/skills/ipynb/scripts/ipynb_tool.py validate path/to/notebook.ipynb
   ```

---

### Procedure 3: Headless Execution & Testing
To run a notebook in CI, background verification, or model evaluation:

```bash
python .agents/skills/ipynb/scripts/ipynb_tool.py execute path/to/notebook.ipynb \
  --timeout 600 \
  --cwd path/to/project_root
```
- Returns exit code `0` on success.
- If a cell throws an exception, returns exit code `2`, prints the exact cell index and ANSI-stripped traceback, and saves execution state up to the failure point.

For parameter sweeps (e.g. changing epochs or batch size dynamically), use [Papermill](./references/execution_headless.md):
```bash
papermill input.ipynb output.ipynb -p batch_size 32 -p lr 0.0001
```

---

### Procedure 4: Git Cleanliness & Commits
Before staging `.ipynb` files to Git, prevent noisy diffs:

1. **Clear execution outputs and counters**:
   ```bash
   python .agents/skills/ipynb/scripts/ipynb_tool.py clear-outputs path/to/notebook.ipynb
   ```
2. Alternatively, configure `nbstripout` globally or per-repo:
   ```bash
   nbstripout install
   ```
   See [Git and Diffs Reference](./references/git_and_diffs.md) for full `.gitattributes` configuration.

---

### Procedure 5: Linting & Formatting with Ruff
Ruff supports `.ipynb` files out-of-the-box:
```bash
# Check notebook for linting errors
ruff check path/to/notebook.ipynb

# Format code cells in-place
ruff format path/to/notebook.ipynb
```
See [Testing and Linting Reference](./references/testing_and_linting.md) for ignoring standard exploratory notebook rules (e.g., `E402`, `F841`).

---

## Reference Guides

- [**nbformat Schema & Programmatic Patterns**](./references/nbformat_cheatsheet.md): Detailed reference for Jupyter schema v4, cell structure, metadata, output types, and raw Python manipulation patterns.
- [**Headless Execution with nbclient & Papermill**](./references/execution_headless.md): Guide to running notebooks headlessly, parameter injection, timeout management, and kernel configuration.
- [**Git Version Control, Diffs, and Sanitization**](./references/git_and_diffs.md): How to use `nbstripout`, `nbdime`, and Jupytext paired percent scripts to keep Git repositories clean.
- [**Testing and Linting Guide**](./references/testing_and_linting.md): How to lint with Ruff, enforce type hints with `nbqa`, and write `pytest` unit tests with `testbook`.
- [**Agent Troubleshooting & Safe Operations**](./references/agent_troubleshooting.md): Specific tips for AI agents to avoid context overflow, navigate tool limitations, and fix corrupted notebooks.

---

## Practical Examples & Resources

- [**Example: Inspect & Edit Workflow**](./examples/workflow_inspect_and_edit.md): Step-by-step example modifying training hyperparameters.
- [**Example: Headless Run Workflow**](./examples/workflow_headless_run.md): Step-by-step example running environment validation.
- [**Sample Notebook**](./examples/sample_notebook.ipynb): Clean reference `.ipynb` conforming to v4 schema.
- [**Ruff Configuration Snippet**](./resources/pyproject_ruff_snippet.toml): Ready-to-use `pyproject.toml` configuration for notebooks.
- [**Gitattributes Sample**](./resources/gitattributes_sample): Ready-to-use `.gitattributes` template for `nbstripout` and `nbdime`.
