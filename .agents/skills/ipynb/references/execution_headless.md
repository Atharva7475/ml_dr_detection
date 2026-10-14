# Headless Notebook Execution Guide

Headless execution runs a Jupyter Notebook from start to finish without launching a browser UI. This is critical for automated testing, continuous integration (CI), batch model training, and AI agent verification.

---

## 1. Execution Engine Comparison

| Tool | Core Strength | Parameter Injection | CLI Available | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **`nbclient`** | Lightweight, direct Python API | Manual / via AST | Via Python script | Fast programmatic runs, custom error recovery |
| **`papermill`** | Parameterization, metrics tracking | Native via cell tags (`parameters`) | `papermill input.ipynb output.ipynb -p lr 0.001` | Data pipelines, ML model sweeps, scheduled jobs |
| **`nbconvert`** | Bundled with Jupyter | None | `jupyter nbconvert --execute --to notebook` | Standard local scripting, exporting to PDF/HTML |

---

## 2. Using `nbclient` (Direct Python Control)

`nbclient` executes notebooks directly with fine-grained control over execution timeouts, working directory, and kernel environment.

### Basic In-Memory Execution
```python
import nbformat
from nbclient import NotebookClient

# Load notebook
nb = nbformat.read("experiment.ipynb", as_version=4)

# Configure client
client = NotebookClient(
    nb,
    timeout=600,             # Max seconds allowed per individual cell
    kernel_name="python3",   # Kernel to execute with
    resources={
        "metadata": {
            "path": "./"     # Set working directory context so relative paths resolve
        }
    }
)

# Execute all cells
try:
    client.execute()
    print("Execution completed with zero errors.")
except Exception as e:
    print(f"Execution failed: {e}")
finally:
    # Save notebook state with populated outputs
    with open("experiment_executed.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
```

### Catching and Diagnosing Errors
When a cell raises an unhandled exception, `nbclient` raises `CellExecutionError`:

```python
from nbclient.exceptions import CellExecutionError

try:
    client.execute()
except CellExecutionError as e:
    # CellExecutionError contains cell index, source, and complete traceback
    print("Failed Cell Execution Traceback:")
    print(e.traceback)
```

---

## 3. Parameterized Execution with `Papermill`

`Papermill` executes notebooks while injecting parameters at runtime.

### Setup:
1. Open the notebook and find the cell defining your default hyperparameters (e.g. `batch_size = 16`, `lr = 1e-3`).
2. Add the tag `"parameters"` to that cell's metadata:
   - In UI: Right sidebar -> Cell Metadata / Tags -> Add `"parameters"`.
   - Programmatically: `cell.metadata["tags"] = ["parameters"]`.

### Execution via CLI:
```bash
papermill notebooks/04_experiment.ipynb results/04_run_32batch.ipynb \
    -p batch_size 32 \
    -p learning_rate 0.0005 \
    -p epochs 10
```

### Execution via Python API:
```python
import papermill as pm

pm.execute_notebook(
    input_path="notebooks/04_experiment.ipynb",
    output_path="results/04_executed.ipynb",
    parameters={
        "batch_size": 32,
        "learning_rate": 0.0005,
        "epochs": 10,
    },
    kernel_name="python3",
    progress_bar=False,
    request_save_on_cell_execute=True,  # Saves partial progress if crashed
)
```

---

## 4. Execution via `jupyter nbconvert` CLI

When running simple headless sweeps in bash/PowerShell without python scripts:

```bash
# Execute in-place (updates the file with outputs)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 path/to/notebook.ipynb

# Execute and output as HTML report
jupyter nbconvert --to html --execute path/to/notebook.ipynb --output report.html

# Allow errors without failing immediately
jupyter nbconvert --to notebook --execute --allow-errors --inplace path/to/notebook.ipynb
```

---

## 5. Best Practices & Pitfalls

1. **Explicit Working Directory (`cwd`)**:
   Notebooks run relative to their own directory by default in JupyterLab, but relative to the caller directory when executed headlessly. Always ensure `path` is set to either the project root or the notebook's folder.
2. **Prevent Infinite Loops with Timeouts**:
   Always specify a timeout (`timeout=300` seconds per cell) so a stuck loop or unresponsive external API does not hang your agent or CI pipeline.
3. **Hardware Acceleration (CUDA/MPS)**:
   Ensure CUDA devices are available in the background process environment. If GPU memory is limited, clear torch cache or restart the kernel between runs.
4. **Handling Visualizations**:
   Add `import matplotlib.pyplot as plt; plt.ioff()` or `%matplotlib inline` to prevent GUI backend popups from freezing headless execution.
