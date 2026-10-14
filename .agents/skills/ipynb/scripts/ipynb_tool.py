#!/usr/bin/env python3
"""
ipynb_tool.py - Comprehensive CLI utility for Jupyter Notebook (.ipynb) operations.
Optimized for AI Agents, CI/CD pipelines, and command-line developers.

Commands:
  outline        Print a token-efficient summary of notebook cells.
  read           Read formatted cell contents (collapses bulky base64 outputs).
  edit           Safely update cell source code by index or tag.
  insert         Insert a new code or markdown cell at an index.
  delete         Delete a cell by index.
  clear-outputs  Strip execution counts and output blobs (like nbstripout).
  execute        Run notebook headlessly via nbclient and capture outputs/tracebacks.
  to-py          Export notebook to standard '# %%' percent format Python script.
  from-py        Import a '# %%' percent format Python script to .ipynb.
  diff           Compare two notebooks or a notebook against a commit/file.
  validate       Validate notebook schema compliance (nbformat v4).
"""

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_nbformat():
    """Load nbformat, raising a helpful error if missing."""
    try:
        import nbformat
        return nbformat
    except ImportError:
        sys.stderr.write("Error: 'nbformat' is required. Install via: pip install nbformat\n")
        sys.exit(1)


def normalize_notebook(nb: Any) -> None:
    """Ensure all cells have unique ID and mandatory schema fields."""
    import uuid
    for cell in nb.cells:
        if "id" not in cell or not cell["id"]:
            cell["id"] = str(uuid.uuid4())[:8]


def read_notebook(filepath: Path) -> Any:
    nbformat = _load_nbformat()
    if not filepath.exists():
        raise FileNotFoundError(f"Notebook not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    normalize_notebook(nb)
    return nb


def write_notebook(nb: Any, filepath: Path) -> None:
    nbformat = _load_nbformat()
    normalize_notebook(nb)
    # Validate before saving
    nbformat.validate(nb)
    with open(filepath, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


# =====================================================================
# 1. OUTLINE
# =====================================================================
def cmd_outline(args: argparse.Namespace) -> int:
    """Print a concise cell outline without output blobs."""
    filepath = Path(args.notebook)
    nb = read_notebook(filepath)
    
    total_cells = len(nb.cells)
    print(f"=== Notebook: {filepath.name} ({total_cells} cells) ===")
    
    for idx, cell in enumerate(nb.cells):
        ctype = cell.cell_type
        exec_count = cell.get("execution_count", " ")
        tags = cell.get("metadata", {}).get("tags", [])
        tag_str = f" [tags: {','.join(tags)}]" if tags else ""
        
        source_lines = cell.source.splitlines()
        num_lines = len(source_lines)
        first_line = source_lines[0].strip() if source_lines else "<empty>"
        if len(first_line) > 70:
            first_line = first_line[:67] + "..."
            
        outputs_count = len(cell.get("outputs", []))
        out_summary = f" [outputs: {outputs_count}]" if outputs_count > 0 else ""
        
        if ctype == "code":
            print(f"[{idx:3d} | code | #{str(exec_count):3s} | {num_lines:2d}L{out_summary}{tag_str}]: {first_line}")
        elif ctype == "markdown":
            print(f"[{idx:3d} | md   |      | {num_lines:2d}L{tag_str}]: {first_line}")
        else:
            print(f"[{idx:3d} | {ctype:4s} |      | {num_lines:2d}L{tag_str}]: {first_line}")
            
    return 0


# =====================================================================
# 2. READ
# =====================================================================
def _summarize_output(out: Dict[str, Any], max_text_lines: int) -> str:
    """Safely summarize an output item without dumping huge base64 strings."""
    out_type = out.get("output_type", "unknown")
    
    if out_type == "stream":
        text = out.get("text", "")
        lines = text.splitlines()
        if len(lines) > max_text_lines:
            truncated = lines[:max_text_lines]
            return "\n".join(truncated) + f"\n... [{len(lines) - max_text_lines} lines truncated]"
        return text.rstrip()
        
    elif out_type in ("execute_result", "display_data"):
        data = out.get("data", {})
        result_parts = []
        for mime, val in data.items():
            if mime.startswith("image/"):
                size = len(val) if isinstance(val, str) else 0
                result_parts.append(f"<{mime} attachment, ~{size // 1024} KB>")
            elif mime == "text/plain":
                if isinstance(val, list):
                    val = "".join(val)
                lines = val.splitlines()
                if len(lines) > max_text_lines:
                    val = "\n".join(lines[:max_text_lines]) + f"\n... [{len(lines) - max_text_lines} lines truncated]"
                result_parts.append(val.rstrip())
            elif mime == "text/html":
                result_parts.append("<text/html rendered output>")
            else:
                result_parts.append(f"<{mime} output>")
        return "\n".join(result_parts)
        
    elif out_type == "error":
        ename = out.get("ename", "Error")
        evalue = out.get("evalue", "")
        tb = out.get("traceback", [])
        # Strip ANSI escape codes from traceback for readability
        ansi_regex = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_tb = [ansi_regex.sub('', line) for line in tb]
        if len(clean_tb) > max_text_lines:
            clean_tb = clean_tb[-max_text_lines:]
        return f"[ERROR: {ename}: {evalue}]\n" + "\n".join(clean_tb)
        
    return f"[{out_type}]"


def cmd_read(args: argparse.Namespace) -> int:
    """Read cell source code and optional sanitized outputs."""
    filepath = Path(args.notebook)
    nb = read_notebook(filepath)
    
    start_idx = args.start if args.start is not None else 0
    end_idx = args.end if args.end is not None else len(nb.cells) - 1
    
    if args.cell is not None:
        start_idx = args.cell
        end_idx = args.cell
        
    if start_idx < 0 or end_idx >= len(nb.cells) or start_idx > end_idx:
        sys.stderr.write(f"Error: Invalid cell index range [{start_idx}, {end_idx}]. Notebook has {len(nb.cells)} cells.\n")
        return 1
        
    for idx in range(start_idx, end_idx + 1):
        cell = nb.cells[idx]
        ctype = cell.cell_type
        tags = cell.get("metadata", {}).get("tags", [])
        tag_str = f" tags={tags}" if tags else ""
        exec_count = cell.get("execution_count", " ")
        
        print(f"--- [CELL {idx} | {ctype.upper()} | exec:{exec_count}{tag_str}] ---")
        if ctype == "code":
            print("```python")
            print(cell.source)
            print("```")
        elif ctype == "markdown":
            print(cell.source)
        else:
            print(cell.source)
            
        if not args.no_outputs and ctype == "code" and cell.get("outputs"):
            print("  --- Output ---")
            for out in cell["outputs"]:
                summary = _summarize_output(out, max_text_lines=args.max_output_lines)
                if summary:
                    print(summary)
        print()
        
    return 0


# =====================================================================
# 3. EDIT CELL
# =====================================================================
def cmd_edit(args: argparse.Namespace) -> int:
    """Safely replace the source of a cell by index or tag."""
    filepath = Path(args.notebook)
    nb = read_notebook(filepath)
    
    target_idx = None
    if args.cell is not None:
        target_idx = args.cell
    elif args.tag:
        for idx, cell in enumerate(nb.cells):
            tags = cell.get("metadata", {}).get("tags", [])
            if args.tag in tags:
                target_idx = idx
                break
        if target_idx is None:
            sys.stderr.write(f"Error: No cell found with tag '{args.tag}'.\n")
            return 1
    else:
        sys.stderr.write("Error: Must specify either --cell <index> or --tag <tag>.\n")
        return 1
        
    if target_idx < 0 or target_idx >= len(nb.cells):
        sys.stderr.write(f"Error: Cell index {target_idx} out of range (0-{len(nb.cells) - 1}).\n")
        return 1
        
    # Get new content
    if args.content is not None:
        new_source = args.content
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            new_source = f.read()
    else:
        sys.stderr.write("Error: Must provide either --content '<text>' or --file <path>.\n")
        return 1
        
    cell = nb.cells[target_idx]
    old_source = cell.source
    cell.source = new_source
    
    # If cell is code and code changed, reset execution count and outputs if requested
    if cell.cell_type == "code" and not args.keep_outputs:
        cell.execution_count = None
        cell.outputs = []
        
    # Save notebook
    out_path = Path(args.output) if args.output else filepath
    write_notebook(nb, out_path)
    
    print(f"Successfully updated Cell {target_idx} in '{out_path}'.")
    if args.diff:
        print("Diff of changes:")
        diff = difflib.unified_diff(
            old_source.splitlines(keepends=True),
            new_source.splitlines(keepends=True),
            fromfile=f"cell_{target_idx}_before",
            tofile=f"cell_{target_idx}_after"
        )
        sys.stdout.writelines(diff)
        
    return 0


# =====================================================================
# 4. INSERT CELL
# =====================================================================
def cmd_insert(args: argparse.Namespace) -> int:
    """Insert a new cell at the given index."""
    nbformat = _load_nbformat()
    filepath = Path(args.notebook)
    nb = read_notebook(filepath)
    
    idx = args.index
    if idx < 0:
        idx = max(0, len(nb.cells) + idx + 1)
    if idx > len(nb.cells):
        idx = len(nb.cells)
        
    # Get content
    if args.content is not None:
        source = args.content
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            source = f.read()
    else:
        source = ""
        
    cell_type = args.type.lower()
    if cell_type == "code":
        new_cell = nbformat.v4.new_code_cell(source=source)
    elif cell_type == "markdown":
        new_cell = nbformat.v4.new_markdown_cell(source=source)
    elif cell_type == "raw":
        new_cell = nbformat.v4.new_raw_cell(source=source)
    else:
        sys.stderr.write(f"Error: Unknown cell type '{cell_type}'. Use code, markdown, or raw.\n")
        return 1
        
    if args.tags:
        new_cell.metadata["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
        
    nb.cells.insert(idx, new_cell)
    
    out_path = Path(args.output) if args.output else filepath
    write_notebook(nb, out_path)
    print(f"Successfully inserted {cell_type} cell at index {idx} in '{out_path}'. Total cells: {len(nb.cells)}.")
    return 0


# =====================================================================
# 5. DELETE CELL
# =====================================================================
def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a cell at the given index."""
    filepath = Path(args.notebook)
    nb = read_notebook(filepath)
    
    idx = args.cell
    if idx < 0 or idx >= len(nb.cells):
        sys.stderr.write(f"Error: Cell index {idx} out of range (0-{len(nb.cells) - 1}).\n")
        return 1
        
    removed = nb.cells.pop(idx)
    out_path = Path(args.output) if args.output else filepath
    write_notebook(nb, out_path)
    print(f"Successfully removed Cell {idx} ({removed.cell_type}) from '{out_path}'. Remaining cells: {len(nb.cells)}.")
    return 0


# =====================================================================
# 6. CLEAR OUTPUTS (nbstripout equivalent)
# =====================================================================
def cmd_clear_outputs(args: argparse.Namespace) -> int:
    """Clear all execution counts and cell outputs."""
    filepath = Path(args.notebook)
    nb = read_notebook(filepath)
    
    cleared_count = 0
    for cell in nb.cells:
        if cell.cell_type == "code":
            if cell.get("outputs") or cell.get("execution_count") is not None:
                cleared_count += 1
            cell.outputs = []
            cell.execution_count = None
            
    out_path = Path(args.output) if args.output else filepath
    write_notebook(nb, out_path)
    print(f"Cleared outputs from {cleared_count} code cells in '{out_path}'.")
    return 0


# =====================================================================
# 7. EXECUTE NOTEBOOK
# =====================================================================
def cmd_execute(args: argparse.Namespace) -> int:
    """Execute a notebook headlessly via nbclient and save outputs."""
    try:
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
    except ImportError:
        sys.stderr.write("Error: 'nbclient' is required for execution. Install via: pip install nbclient\n")
        return 1
        
    filepath = Path(args.notebook)
    nb = read_notebook(filepath)
    
    cwd = args.cwd if args.cwd else str(filepath.parent.resolve())
    timeout = args.timeout if args.timeout is not None else 600
    kernel_name = args.kernel if args.kernel else "python3"
    
    print(f"Executing '{filepath.name}' (kernel='{kernel_name}', timeout={timeout}s, cwd='{cwd}')...")
    
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": cwd}},
    )
    
    out_path = Path(args.output) if args.output else filepath
    exit_code = 0
    try:
        client.execute()
        print("Execution completed successfully!")
    except CellExecutionError as e:
        sys.stderr.write(f"\nExecution failed with CellExecutionError:\n{str(e)}\n")
        exit_code = 2
    except Exception as e:
        sys.stderr.write(f"\nExecution failed with error: {str(e)}\n")
        exit_code = 1
    finally:
        write_notebook(nb, out_path)
        print(f"Notebook state saved to '{out_path}'.")
        
    return exit_code


# =====================================================================
# 8. TO-PY (Export to % percent format)
# =====================================================================
def cmd_to_py(args: argparse.Namespace) -> int:
    """Export notebook to a standard '# %%' percent format Python script."""
    filepath = Path(args.notebook)
    nb = read_notebook(filepath)
    
    lines = [
        f'# %% [markdown]',
        f'# # Converted from {filepath.name}',
        f'#',
    ]
    
    for idx, cell in enumerate(nb.cells):
        ctype = cell.cell_type
        tags = cell.get("metadata", {}).get("tags", [])
        tag_line = f" [tags: {','.join(tags)}]" if tags else ""
        
        if ctype == "code":
            lines.append(f"\n# %% [code]{tag_line}")
            lines.append(cell.source.rstrip())
        elif ctype == "markdown":
            lines.append(f"\n# %% [markdown]{tag_line}")
            for md_line in cell.source.splitlines():
                lines.append(f"# {md_line}" if md_line.strip() else "#")
        else:
            lines.append(f"\n# %% [{ctype}]{tag_line}")
            for r_line in cell.source.splitlines():
                lines.append(f"# {r_line}" if r_line.strip() else "#")
                
    content = "\n".join(lines) + "\n"
    
    out_path = Path(args.output) if args.output else filepath.with_suffix(".py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Successfully exported {len(nb.cells)} cells to '{out_path}'.")
    return 0


# =====================================================================
# 9. FROM-PY (Import from % percent format)
# =====================================================================
def cmd_from_py(args: argparse.Namespace) -> int:
    """Import a '# %%' percent format Python script into an .ipynb file."""
    nbformat = _load_nbformat()
    py_path = Path(args.python_file)
    if not py_path.exists():
        sys.stderr.write(f"Error: File not found: {py_path}\n")
        return 1
        
    with open(py_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    cells = []
    current_type = "code"
    current_lines = []
    current_tags = []
    
    cell_marker = re.compile(r"^#\s*%%\s*(\[(\w+)\])?(.*)$")
    
    def flush_cell():
        nonlocal current_lines, current_type, current_tags
        if not current_lines and not cells:
            return
        text = "".join(current_lines).strip("\n")
        if current_type == "markdown":
            # Strip leading '# ' or '#' from markdown lines
            cleaned_lines = []
            for l in text.splitlines():
                if l.startswith("# "):
                    cleaned_lines.append(l[2:])
                elif l.startswith("#"):
                    cleaned_lines.append(l[1:])
                else:
                    cleaned_lines.append(l)
            text = "\n".join(cleaned_lines)
            cell = nbformat.v4.new_markdown_cell(source=text)
        else:
            cell = nbformat.v4.new_code_cell(source=text)
            
        if current_tags:
            cell.metadata["tags"] = current_tags
        cells.append(cell)
        current_lines = []
        current_tags = []
        current_type = "code"
        
    for line in lines:
        match = cell_marker.match(line.rstrip())
        if match:
            flush_cell()
            ctype = match.group(2)
            extra = match.group(3) or ""
            if ctype:
                current_type = ctype.lower()
            if "tags:" in extra:
                tag_match = re.search(r"tags:\s*([^\]]+)", extra)
                if tag_match:
                    current_tags = [t.strip() for t in tag_match.group(1).split(",") if t.strip()]
        else:
            current_lines.append(line)
            
    flush_cell()
    
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    
    out_path = Path(args.output) if args.output else py_path.with_suffix(".ipynb")
    write_notebook(nb, out_path)
    print(f"Successfully converted '{py_path.name}' to '{out_path}' with {len(cells)} cells.")
    return 0


# =====================================================================
# 10. DIFF
# =====================================================================
def cmd_diff(args: argparse.Namespace) -> int:
    """Compare two notebooks cell-by-cell."""
    path1 = Path(args.nb1)
    path2 = Path(args.nb2)
    
    nb1 = read_notebook(path1)
    nb2 = read_notebook(path2)
    
    diff_found = False
    max_len = max(len(nb1.cells), len(nb2.cells))
    
    print(f"=== Comparing {path1.name} vs {path2.name} ===")
    
    for i in range(max_len):
        c1 = nb1.cells[i] if i < len(nb1.cells) else None
        c2 = nb2.cells[i] if i < len(nb2.cells) else None
        
        if c1 is None:
            print(f"\n[Cell {i}]: Added in {path2.name} ({c2.cell_type})")
            print("+" + "\n+".join(c2.source.splitlines()))
            diff_found = True
            continue
            
        if c2 is None:
            print(f"\n[Cell {i}]: Deleted from {path2.name} ({c1.cell_type})")
            print("-" + "\n-".join(c1.source.splitlines()))
            diff_found = True
            continue
            
        if c1.cell_type != c2.cell_type or c1.source != c2.source:
            diff_found = True
            print(f"\n[Cell {i}]: Modified (type: {c1.cell_type} -> {c2.cell_type})")
            diff = difflib.unified_diff(
                c1.source.splitlines(keepends=True),
                c2.source.splitlines(keepends=True),
                fromfile=f"{path1.name}#cell_{i}",
                tofile=f"{path2.name}#cell_{i}"
            )
            sys.stdout.writelines(diff)
            
    if not diff_found:
        print("Notebooks are identical in cell types and source code.")
        return 0
    return 1


# =====================================================================
# 11. VALIDATE
# =====================================================================
def cmd_validate(args: argparse.Namespace) -> int:
    """Validate notebook schema."""
    nbformat = _load_nbformat()
    filepath = Path(args.notebook)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        nbformat.validate(nb)
        print(f"VALID: '{filepath}' conforms to nbformat v4 schema ({len(nb.cells)} cells).")
        return 0
    except Exception as e:
        sys.stderr.write(f"INVALID: '{filepath}' failed validation: {e}\n")
        return 1


# =====================================================================
# MAIN DISPATCHER
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ipynb_tool: Powerful CLI for Jupyter Notebook (.ipynb) operations."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    
    # outline
    p_outline = subparsers.add_parser("outline", help="Print cell overview and statistics")
    p_outline.add_argument("notebook", help="Path to .ipynb file")
    p_outline.set_defaults(func=cmd_outline)
    
    # read
    p_read = subparsers.add_parser("read", help="Read formatted cell code and text")
    p_read.add_argument("notebook", help="Path to .ipynb file")
    p_read.add_argument("--cell", type=int, help="Read specific cell index only")
    p_read.add_argument("--start", type=int, help="Start cell index")
    p_read.add_argument("--end", type=int, help="End cell index")
    p_read.add_argument("--no-outputs", action="store_true", help="Omit execution outputs")
    p_read.add_argument("--max-output-lines", type=int, default=15, help="Max output lines per cell")
    p_read.set_defaults(func=cmd_read)
    
    # edit
    p_edit = subparsers.add_parser("edit", help="Edit a cell source code")
    p_edit.add_argument("notebook", help="Path to .ipynb file")
    p_edit.add_argument("--cell", type=int, help="Cell index to modify")
    p_edit.add_argument("--tag", type=str, help="Cell tag to match")
    p_edit.add_argument("--content", type=str, help="New source string")
    p_edit.add_argument("--file", type=str, help="File containing new source")
    p_edit.add_argument("--output", type=str, help="Save destination (default: in-place)")
    p_edit.add_argument("--keep-outputs", action="store_true", help="Do not clear outputs for this cell")
    p_edit.add_argument("--diff", action="store_true", help="Print diff of changes")
    p_edit.set_defaults(func=cmd_edit)
    
    # insert
    p_insert = subparsers.add_parser("insert", help="Insert a new cell")
    p_insert.add_argument("notebook", help="Path to .ipynb file")
    p_insert.add_argument("--index", type=int, required=True, help="Index where new cell is placed")
    p_insert.add_argument("--type", choices=["code", "markdown", "raw"], default="code", help="Cell type")
    p_insert.add_argument("--content", type=str, help="Cell source content")
    p_insert.add_argument("--file", type=str, help="File containing cell source")
    p_insert.add_argument("--tags", type=str, help="Comma-separated cell tags")
    p_insert.add_argument("--output", type=str, help="Save destination (default: in-place)")
    p_insert.set_defaults(func=cmd_insert)
    
    # delete
    p_delete = subparsers.add_parser("delete", help="Delete a cell")
    p_delete.add_argument("notebook", help="Path to .ipynb file")
    p_delete.add_argument("--cell", type=int, required=True, help="Cell index to delete")
    p_delete.add_argument("--output", type=str, help="Save destination (default: in-place)")
    p_delete.set_defaults(func=cmd_delete)
    
    # clear-outputs
    p_clear = subparsers.add_parser("clear-outputs", help="Clear all execution outputs and counts")
    p_clear.add_argument("notebook", help="Path to .ipynb file")
    p_clear.add_argument("--output", type=str, help="Save destination (default: in-place)")
    p_clear.set_defaults(func=cmd_clear_outputs)
    
    # execute
    p_exec = subparsers.add_parser("execute", help="Execute notebook headlessly")
    p_exec.add_argument("notebook", help="Path to .ipynb file")
    p_exec.add_argument("--timeout", type=int, default=600, help="Per-cell timeout in seconds")
    p_exec.add_argument("--cwd", type=str, help="Working directory context for execution")
    p_exec.add_argument("--kernel", type=str, default="python3", help="Jupyter kernel name")
    p_exec.add_argument("--output", type=str, help="Save executed notebook (default: in-place)")
    p_exec.set_defaults(func=cmd_execute)
    
    # to-py
    p_topy = subparsers.add_parser("to-py", help="Convert .ipynb to '# %%' percent Python script")
    p_topy.add_argument("notebook", help="Path to .ipynb file")
    p_topy.add_argument("--output", type=str, help="Destination .py path")
    p_topy.set_defaults(func=cmd_to_py)
    
    # from-py
    p_frompy = subparsers.add_parser("from-py", help="Convert '# %%' percent Python script to .ipynb")
    p_frompy.add_argument("python_file", help="Path to .py file")
    p_frompy.add_argument("--output", type=str, help="Destination .ipynb path")
    p_frompy.set_defaults(func=cmd_from_py)
    
    # diff
    p_diff = subparsers.add_parser("diff", help="Diff two notebooks cell-by-cell")
    p_diff.add_argument("nb1", help="First notebook")
    p_diff.add_argument("nb2", help="Second notebook")
    p_diff.set_defaults(func=cmd_diff)
    
    # validate
    p_val = subparsers.add_parser("validate", help="Validate notebook schema")
    p_val.add_argument("notebook", help="Path to .ipynb file")
    p_val.set_defaults(func=cmd_validate)
    
    parsed_args = parser.parse_args()
    code = parsed_args.func(parsed_args)
    sys.exit(code if code is not None else 0)


if __name__ == "__main__":
    main()
