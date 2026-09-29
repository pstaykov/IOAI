"""Execute notebooks with nbclient and log results.

usage: run.py [--timeout S] [--env K=V ...] nb1.ipynb nb2.ipynb ...
Runs each notebook with cwd = scratch/run/<Area> so downloads are shared per area.
"""
import sys, os, json, time, re, argparse, traceback
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = Path(__file__).parent
SCRATCH = HERE / "_out"
REPO = HERE.resolve().parents[1]          # tools/notebook_build -> repo root
AREAS = ["TraditionalML", "DL", "CompuerVision", "NLPAudio"]


def area_of(p):
    for a in AREAS:
        if a in Path(p).parts: return a
    return "misc"


def summarize_outputs(nb, pattern=r"(?i)(leaderboard|score|accuracy|acc|f1|loss|check|pass|ok|✓|elapsed|time)"):
    hits = []
    for c in nb.cells:
        if c.cell_type != "code": continue
        for o in c.get("outputs", []):
            txt = o.get("text") or "".join(o.get("data", {}).get("text/plain", ""))
            for line in str(txt).splitlines():
                if re.search(pattern, line): hits.append(line.strip()[:160])
    return hits[-25:]


def run(path, timeout, env):
    path = Path(path)
    nb = nbformat.read(path, as_version=4)
    cwd = SCRATCH / "run" / area_of(path)
    cwd.mkdir(parents=True, exist_ok=True)
    old = dict(os.environ)
    os.environ.update(env)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    t0 = time.time()
    status, err = "ok", ""
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(cwd)}})
    try:
        client.execute()
    except CellExecutionError as e:
        status = "error"
        err = str(e)[-3000:]
    except Exception as e:
        status = "error"
        err = traceback.format_exc()[-3000:]
    finally:
        os.environ.clear(); os.environ.update(old)
    dt = time.time() - t0
    parts = path.parts[path.parts.index(area_of(path)) + 1:] if area_of(path) in path.parts else (path.name,)
    tag = "filled_" if "test" in path.parts else ""
    out = SCRATCH / "executed" / area_of(path) / (tag + "__".join(parts))
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    rec = {"notebook": str(path), "status": status, "seconds": round(dt, 1), "error": err,
           "highlights": summarize_outputs(nb), "executed": str(out)}
    with open(SCRATCH / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("nbs", nargs="+")
    a = ap.parse_args()
    env = dict(kv.split("=", 1) for kv in a.env)
    for p in a.nbs:
        r = run(p, a.timeout, env)
        print(f"[{r['status']}] {r['seconds']:7.1f}s  {p}")
        for h in r["highlights"][-12:]: print("     ", h)
        if r["status"] != "ok": print(r["error"][-2500:])
        sys.stdout.flush()
