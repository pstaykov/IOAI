"""Build all notebooks from build/src into the repo (+ filled test copies into scratch)."""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nbsrc import build_guided, build_drill, build_exercise

HERE = Path(__file__).parent
REPO = HERE.resolve().parents[1]          # tools/notebook_build -> repo root
TEST = HERE / "_out" / "test"


def title_of(src):
    for line in Path(src).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^#\s+(.+)$", line)
        if m: return m.group(1)
    return Path(src).stem


def main(filters):
    for src in sorted((HERE / "src").rglob("*.src")):
        area = src.parent.name
        name, kind = src.stem.rsplit(".", 1)
        key = f"{area}/{name}.{kind}"
        if filters and not any(f in key for f in filters): continue
        t = title_of(src)
        if kind == "guided":
            n = build_guided(src, REPO / area / "guided" / f"{name}.ipynb", TEST / area / "guided" / f"{name}.ipynb",
                             REPO / area / "solutions" / "guided" / f"{name}.md", t)
            print(f"{key}: {n} gaps")
        elif kind == "drill":
            n = build_drill(src, REPO / area / "drills" / f"{name}.ipynb", TEST / area / "drills" / f"{name}.ipynb",
                            REPO / area / "solutions" / "drills" / f"{name}.py", t)
            print(f"{key}: {n} drill cells")
        elif kind == "exercise":
            build_exercise(src, REPO / area / "exercises" / f"{name}.ipynb",
                           REPO / area / "solutions" / "exercises" / f"{name}.ipynb")
            print(f"{key}: task + solution")
        else:
            raise ValueError(src)


if __name__ == "__main__":
    main(sys.argv[1:])
