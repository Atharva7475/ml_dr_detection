"""
test_ipynb_tool.py - Automated tests for ipynb_tool.py CLI
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import nbformat


class TestIpynbTool(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.nb_path = Path(self.temp_dir) / "test_sample.ipynb"
        
        # Create a test notebook
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_markdown_cell("# Heading 1"),
            nbformat.v4.new_code_cell("x = 10\nprint(x)", execution_count=1),
            nbformat.v4.new_code_cell("y = x * 2\nprint(y)", execution_count=2),
        ]
        # Add output to cell 1
        nb.cells[1].outputs = [
            nbformat.v4.new_output("stream", name="stdout", text="10\n")
        ]
        with open(self.nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
            
        self.script = Path(__file__).parent / "ipynb_tool.py"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def run_tool(self, *args):
        cmd = [sys.executable, str(self.script)] + list(args)
        return subprocess.run(cmd, capture_output=True, text=True, check=True)

    def test_outline(self):
        res = self.run_tool("outline", str(self.nb_path))
        self.assertIn("3 cells", res.stdout)
        self.assertIn("Heading 1", res.stdout)
        self.assertIn("x = 10", res.stdout)

    def test_read_cell(self):
        res = self.run_tool("read", str(self.nb_path), "--cell", "1")
        self.assertIn("CELL 1 | CODE", res.stdout)
        self.assertIn("x = 10", res.stdout)
        self.assertIn("10", res.stdout)

    def test_edit_cell(self):
        self.run_tool("edit", str(self.nb_path), "--cell", "1", "--content", "x = 99\nprint(x)")
        # Verify content was updated
        with open(self.nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        self.assertEqual(nb.cells[1].source, "x = 99\nprint(x)")
        self.assertEqual(nb.cells[1].outputs, [])

    def test_insert_and_delete_cell(self):
        # Insert
        self.run_tool("insert", str(self.nb_path), "--index", "1", "--type", "markdown", "--content", "## Subheading")
        with open(self.nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        self.assertEqual(len(nb.cells), 4)
        self.assertEqual(nb.cells[1].cell_type, "markdown")
        self.assertEqual(nb.cells[1].source, "## Subheading")

        # Delete
        self.run_tool("delete", str(self.nb_path), "--cell", "1")
        with open(self.nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        self.assertEqual(len(nb.cells), 3)

    def test_clear_outputs(self):
        self.run_tool("clear-outputs", str(self.nb_path))
        with open(self.nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        for c in nb.cells:
            if c.cell_type == "code":
                self.assertEqual(c.outputs, [])
                self.assertIsNone(c.execution_count)

    def test_to_py_and_from_py_roundtrip(self):
        py_path = Path(self.temp_dir) / "roundtrip.py"
        nb_restored = Path(self.temp_dir) / "restored.ipynb"

        self.run_tool("to-py", str(self.nb_path), "--output", str(py_path))
        self.assertTrue(py_path.exists())
        self.assertIn("# %% [code]", py_path.read_text(encoding="utf-8"))

        self.run_tool("from-py", str(py_path), "--output", str(nb_restored))
        self.assertTrue(nb_restored.exists())
        
        # Validate restored
        res = self.run_tool("validate", str(nb_restored))
        self.assertIn("VALID", res.stdout)

    def test_diff(self):
        nb2 = Path(self.temp_dir) / "test_sample_2.ipynb"
        shutil.copyfile(self.nb_path, nb2)
        # Edit nb2
        self.run_tool("edit", str(nb2), "--cell", "2", "--content", "y = x * 100")
        
        cmd = [sys.executable, str(self.script), "diff", str(self.nb_path), str(nb2)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)  # diff returns 1 when differences found
        self.assertIn("Modified", proc.stdout)
        self.assertIn("y = x * 100", proc.stdout)

    def test_validate(self):
        res = self.run_tool("validate", str(self.nb_path))
        self.assertIn("VALID", res.stdout)


if __name__ == "__main__":
    unittest.main()
