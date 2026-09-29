"""Tiny notebook source format -> .ipynb variants.

Source format (one file per notebook):
    @@md [only=task|sol]         markdown cell
    @@code [only=task|sol] [drill=ID]   code cell
Inside code cells (guided notebooks):
    @@gap <hint>                 start of a gap; following lines = answer
    @@as                         following single line = placeholder shown to student (contains ____)
    @@end
Variants:
    guided   -> student (TODO + placeholder), filled (answers), solutions markdown
    drill    -> student (drill cells emptied), filled (reference code), reference .py
    exercise -> task notebook (only!=sol), solution notebook (only!=task)
"""
import json, re, textwrap
from pathlib import Path

KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}


def parse(path):
    text = Path(path).read_text(encoding="utf-8")
    cells, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^@@(md|code)\b(.*)$", line)
        if m:
            if cur: cells.append(cur)
            attrs = dict(re.findall(r"(\w+)=(\S+)", m.group(2)))
            cur = {"type": m.group(1), "attrs": attrs, "lines": []}
        else:
            if cur is None:
                if line.strip(): raise ValueError(f"{path}: text before first cell: {line!r}")
                continue
            cur["lines"].append(line)
    if cur: cells.append(cur)
    for c in cells:  # trim blank lines at both ends
        while c["lines"] and not c["lines"][-1].strip(): c["lines"].pop()
        while c["lines"] and not c["lines"][0].strip(): c["lines"].pop(0)
    return cells


def split_gaps(lines, counter):
    """Return (student_lines, filled_lines, gaps[(id, hint, answer, placeholder)])."""
    stu, fil, gaps = [], [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)@@gap\s+(.*)$", line)
        if not m:
            if line.strip().startswith("@@"): raise ValueError(f"stray marker {line!r}")
            stu.append(line); fil.append(line); i += 1; continue
        indent, hint = m.group(1), m.group(2).strip()
        answer, placeholder = [], None
        i += 1
        while not re.match(r"^\s*@@as\s*$", lines[i]):
            answer.append(lines[i]); i += 1
        i += 1
        placeholder = []
        while not re.match(r"^\s*@@end\s*$", lines[i]):
            placeholder.append(lines[i]); i += 1
        i += 1
        counter[0] += 1
        gid = f"g{counter[0]}"
        if not any("____" in p for p in placeholder):
            raise ValueError(f"gap {gid} placeholder lacks ____: {placeholder}")
        ph_indent = re.match(r"^(\s*)", placeholder[0]).group(1)
        stu.append(f"{ph_indent}# TODO({gid}): {hint}")
        stu.extend(placeholder)
        fil.append(f"{ph_indent}# ({gid}) {hint}")
        fil.extend(answer)
        gaps.append((gid, hint, answer, placeholder))
    return stu, fil, gaps


def _cell(kind, src, meta=None):
    src_lines = [l + "\n" for l in src.split("\n")] if src else []
    if src_lines: src_lines[-1] = src_lines[-1].rstrip("\n")
    c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": meta or {}, "source": src_lines}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def write_nb(cells, path):
    nb = {"cells": cells, "metadata": {"kernelspec": KERNEL, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"c{i:03d}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def build_guided(src, out_student, out_filled, out_solutions_md, title):
    cells = parse(src)
    counter = [0]
    stu_cells, fil_cells, all_gaps = [], [], []
    for c in cells:
        body = "\n".join(c["lines"])
        if c["type"] == "md":
            stu_cells.append(_cell("md", body)); fil_cells.append(_cell("md", body)); continue
        s, f, g = split_gaps(c["lines"], counter)
        all_gaps.extend(g)
        stu_cells.append(_cell("code", "\n".join(s)))
        fil_cells.append(_cell("code", "\n".join(f)))
    write_nb(stu_cells, out_student)
    write_nb(fil_cells, out_filled)
    md = [f"# Solutions: {title}", "",
          "One block per `# TODO(gN)` gap. Each block is the full replacement for the placeholder line(s).", ""]
    for gid, hint, ans, ph in all_gaps:
        md += [f"## {gid}: {hint}", "", "Placeholder:", "```python", *[p.strip() for p in ph], "```",
               "Answer:", "```python", *textwrap.dedent("\n".join(ans)).split("\n"), "```", ""]
    Path(out_solutions_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_solutions_md).write_text("\n".join(md), encoding="utf-8")
    return len(all_gaps)


def build_drill(src, out_student, out_filled, out_ref_py, title):
    cells = parse(src)
    stu_cells, fil_cells, ref = [], [], [f"# Reference implementation for drill: {title}",
                                          "# Percent-format cells; each block fills the drill cell with the same id.", ""]
    n = 0
    last_md_title = ""
    for c in cells:
        body = "\n".join(c["lines"])
        if c["type"] == "md":
            m = re.search(r"^#+\s*(.+)$", body, re.M)
            if m: last_md_title = m.group(1)
            stu_cells.append(_cell("md", body)); fil_cells.append(_cell("md", body)); continue
        if "@@gap" in body: raise ValueError("gaps not allowed in drills")
        did = c["attrs"].get("drill")
        if did:
            n += 1
            stu_cells.append(_cell("code", "", {"tags": [f"drill:{did}"]}))
            fil_cells.append(_cell("code", body, {"tags": [f"drill:{did}"]}))
            ref += [f"# %% [{did}] {last_md_title}", body, ""]
        else:
            stu_cells.append(_cell("code", body)); fil_cells.append(_cell("code", body))
    write_nb(stu_cells, out_student)
    write_nb(fil_cells, out_filled)
    Path(out_ref_py).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ref_py).write_text("\n".join(ref), encoding="utf-8")
    return n


def build_exercise(src, out_task, out_solution):
    cells = parse(src)
    task, sol = [], []
    for c in cells:
        body = "\n".join(c["lines"])
        if "@@gap" in body: raise ValueError("gaps not allowed in exercises")
        only = c["attrs"].get("only")
        cell = _cell(c["type"], body)
        if only != "sol": task.append(cell)
        if only != "task": sol.append(json.loads(json.dumps(cell)))
    write_nb(task, out_task)
    write_nb(sol, out_solution)
