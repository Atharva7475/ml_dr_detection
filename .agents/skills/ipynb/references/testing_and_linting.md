# Linting, Formatting, and Testing Notebooks

Modern data science and ML engineering require notebooks to be held to the same software engineering standards as Python production packages.

---

## 1. Fast Linting and Formatting with Ruff

[Ruff](https://astral.sh/ruff) provides built-in, native support for Jupyter Notebook (`.ipynb`) files. It parses code cells while preserving Markdown cells and IPython magic commands (`%timeit`, `!pip`, etc.).

### Command Line Usage
```bash
# Lint code cells across all notebooks
ruff check notebook.ipynb
ruff check notebooks/

# Automatically fix linting violations in-place
ruff check --fix notebook.ipynb

# Format code cells (Black-compatible style)
ruff format notebook.ipynb
```

### Tailoring Rules for Notebooks (`pyproject.toml`)
In notebooks, top-level imports may occur mid-document, or exploratory variables may appear unused. Suppress these non-issues using `per-file-ignores`:

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B"]

[tool.ruff.lint.per-file-ignores]
# Specific rules to ignore for interactive notebooks
"*.ipynb" = [
    "E402",  # Module level import not at top of file
    "F841",  # Local variable name assigned but never used
    "F401",  # Module imported but unused (common for exploratory imports)
    "B018",  # Found useless expression (common for final cell outputs like df)
]
```

---

## 2. Running Legacy Linters with `nbqa`

If your project mandates tools that don't natively understand `.ipynb` (e.g. `mypy`, `flake8`, `pylint`, `black`, `isort`), use **`nbqa`**:

```bash
pip install nbqa
```

### Usage
Prefix any linter or formatter with `nbqa`:

```bash
# Run flake8 on notebook cells
nbqa flake8 notebook.ipynb

# Run mypy type checking
nbqa mypy notebook.ipynb

# Sort imports across cells with isort
nbqa isort notebook.ipynb

# Format cells with Black
nbqa black notebook.ipynb
```

---

## 3. Unit Testing Notebooks with `testbook`

`testbook` allows you to write standard `pytest` unit tests that execute and assert notebook functions, variables, and cell outputs programmatically.

### Installation
```bash
pip install testbook pytest
```

### Writing Tests (`tests/test_analysis_notebook.py`)
```python
from testbook import testbook

# Test with full headless notebook execution
@testbook('notebooks/04_experiment.ipynb', execute=True)
def test_experiment_notebook(tb):
    # 1. Assert that a function defined in the notebook works
    compute_loss = tb.ref('compute_loss')
    loss_val = compute_loss(pred=0.8, target=1.0)
    assert loss_val > 0.0

    # 2. Check variable state after execution
    final_acc = tb.ref('final_accuracy')
    assert final_acc >= 0.85

    # 3. Assert cell stdout output (e.g., cell index 5)
    cell_output = tb.cell_output_text(5)
    assert "Epoch 10/10 complete" in cell_output
```

### Executing Specific Cell Ranges (Faster Unit Tests)
If executing the entire notebook is too slow (e.g., deep learning training loops), execute only setup/helper cells:

```python
# Execute only cells 0 through 4 (skipping the training loop)
@testbook('notebooks/04_experiment.ipynb', execute=range(0, 5))
def test_data_loader_setup(tb):
    dataset = tb.ref('train_dataset')
    assert len(dataset) > 0
```

### Running with Pytest
```bash
pytest tests/test_analysis_notebook.py -v
```
