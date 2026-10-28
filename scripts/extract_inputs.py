"""Extract benchmark inputs from Trinity's Rust optimizer tests.

Each `llama_*`/`falcon_*` test in trinity/optimizer/tests/*.rs carries a
shape table, an input IR string, the equality-saturation iteration limit, and
the extraction bounds (max cost, max kernels).  We write them out as:

  inputs/<workload>_<model>.ir      the IR, verbatim
  inputs/<workload>_<model>.shapes  ((name dim...) ...) as an s-expression
  inputs/manifest.sexp              one plist per instance

Usage: python scripts/extract_inputs.py [path/to/trinity]
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRINITY = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "trinity"
TESTS = TRINITY / "optimizer" / "tests"

TEST_RE = re.compile(r"#\[test\]\s*fn\s+(\w+)\s*\(\)\s*\{")
SHAPE_RE = re.compile(r'\(\s*"([^"]+)"\s*,\s*vec!\[([^\]]*)\]\s*\)')
EXPR_RE = re.compile(r'let\s+expr\s*=\s*"(.*?)";', re.S)
ITER_RE = re.compile(r"run_until_saturated\(\s*expr\s*,\s*rules\(\)\s*,\s*(\d+)\s*,?\s*\)")
# Trinity's gqa_vanilla tests reuse the vanilla IR verbatim with GQA shapes, so
# their loop bounds (4096 columns, 32 heads) disagree with the K/V tensors
# (1024 columns, 8 heads). We leave them out.
SKIP = {"gqa_vanilla"}

def split_tests(src):
    """Yield (name, body) for every #[test] function."""
    starts = [(m.group(1), m.end()) for m in TEST_RE.finditer(src)]
    for i, (name, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(src)
        yield name, src[start:end]


def main():
    out = ROOT / "inputs"
    out.mkdir(exist_ok=True)
    manifest = []
    for rs in sorted(TESTS.glob("*.rs")):
        workload = rs.stem
        if workload in SKIP:
            continue
        src = rs.read_text()
        for name, body in split_tests(src):
            model = name.split("_")[0]
            if model not in ("llama", "falcon"):
                continue
            expr = EXPR_RE.search(body)
            shapes_start = body.find("setup_shape_tracker")
            shapes_end = body.find("]);", shapes_start)
            if not expr or shapes_start < 0:
                print(f"skip {rs.name}:{name}: no expr/shapes", file=sys.stderr)
                continue
            shapes = SHAPE_RE.findall(body[shapes_start:shapes_end])
            it = ITER_RE.search(body)
            stem = f"{workload}_{model}"
            (out / f"{stem}.ir").write_text(expr.group(1).strip() + "\n")
            with open(out / f"{stem}.shapes", "w") as f:
                f.write("(\n")
                for tname, dims in shapes:
                    dims = " ".join(d.strip() for d in dims.split(",") if d.strip())
                    f.write(f'  ("{tname}" {dims})\n')
                f.write(")\n")
            manifest.append(
                f'(:name "{stem}" :workload "{workload}" :model "{model}"'
                f" :iter {it.group(1) if it else 8})"
            )
            print(f"{stem}: {len(shapes)} tensors, iter={it.group(1) if it else '?'}")
    (out / "manifest.sexp").write_text("(\n " + "\n ".join(manifest) + ")\n")


if __name__ == "__main__":
    main()
