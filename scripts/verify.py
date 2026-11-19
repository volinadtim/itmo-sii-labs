"""Execute the notebook and validate its current kernel state."""

from pathlib import Path
import io
import tokenize
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "lab3/Lab3_Danilov_Pivovarov.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
template = nbformat.read(ROOT / "materials/Lab3.ipynb", as_version=4)
text = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]
assert all(
    cell.source in text for cell in template.cells if cell.cell_type == "markdown"
)
for cell in notebook.cells:
    if cell.cell_type == "code":
        assert not any(
            t.type == tokenize.COMMENT
            for t in tokenize.generate_tokens(io.StringIO(cell.source).readline)
        )
count = len(notebook.cells)
notebook.cells.append(
    nbformat.v4.new_code_cell((ROOT / "scripts/check_lab3.py").read_text())
)
client = NotebookClient(
    notebook,
    timeout=1200,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT / "lab3")}},
)
client.execute()
for output in notebook.cells[-1].outputs:
    if output.output_type == "stream":
        print(output.text.strip())
notebook.cells = notebook.cells[:count]
code = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in code)
assert not any(
    output.output_type == "error" for cell in code for output in cell.outputs
)
assert (
    sum(
        "image/png" in output.get("data", {})
        for cell in code
        for output in cell.outputs
    )
    >= 5
)
nbformat.write(notebook, path)
print(
    f"PASS: {len(code)} code cells executed, 5 required figures present, assignment text preserved, no code comments."
)
