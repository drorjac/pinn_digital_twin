"""Build notebooks from percent-format sources in notebooks_src/ and execute them.

    python tools/build.py              # build + execute all
    python tools/build.py 03 05        # only notebooks whose name starts with these prefixes
    python tools/build.py --no-exec    # build without executing
"""

import re
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "notebooks_src", ROOT / "notebooks"


def parse(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "md":
            body = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif body.strip():
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def build(path, execute=True):
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(path.read_text())
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = OUT / (path.stem + ".ipynb")
    if execute:
        t0 = time.time()
        ExecutePreprocessor(timeout=7200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(OUT)}})
        print(f"  executed in {time.time() - t0:.0f}s")
    nbformat.write(nb, out)
    print(f"  -> {out.relative_to(ROOT)}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    execute = "--no-exec" not in sys.argv
    for p in sorted(SRC.glob("*.py")):
        if args and not any(p.name.startswith(a) for a in args):
            continue
        print(p.name)
        build(p, execute)
