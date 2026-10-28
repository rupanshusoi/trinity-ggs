"""Check optimized programs against the input program by running both.

A small numpy interpreter for Trinity's tile IR: tensors live in a global
memory, loops of every kind run sequentially, and symbolic tile sizes default
to 64. Each program starts from the same random inputs (zeros for everything
else), and we compare every output tensor and every input tensor it writes
(e.g. the KV cache).

Usage:
  python scripts/check_equivalence.py INSTANCE [--program FILE...]
  python scripts/check_equivalence.py --original FILE --shapes FILE --program FILE...

By default checks our optimized program, results/INSTANCE.ir.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TILE = 64


def parse(text):
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    pos = 0

    def atom(tok):
        try:
            return int(tok)
        except ValueError:
            try:
                return float(tok)
            except ValueError:
                return tok

    def read():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            items = []
            while tokens[pos] != ")":
                items.append(read())
            pos += 1
            # (fulltile) is the same as fulltile
            return items[0] if len(items) == 1 and isinstance(items[0], str) else items
        return atom(tok)

    return read()


def read_shapes(path):
    shapes = {}
    for name, dims in re.findall(r'\("([^"]+)"((?:\s+\d+)*)\s*\)', path.read_text()):
        shapes[name] = tuple(int(d) for d in dims.split())
    return shapes


def walk(term):
    yield term
    if isinstance(term, list):
        for t in term[1:]:
            yield from walk(t)


def bases(term, kind):
    """Names used under (KIND name) in TERM, with fused names split."""
    names = set()
    for t in walk(term):
        if isinstance(t, list) and t[0] == kind and len(t) == 2:
            names.update(t[1].split(","))
    return names


def written_names(term):
    names = set()
    for t in walk(term):
        if isinstance(t, list) and t[0] == "store":
            names.update(t[1][1].split(","))
    return names


class Interpreter:
    def __init__(self, shapes, inputs):
        self.mem = {name: np.zeros(dims) for name, dims in shapes.items()}
        for name, value in inputs.items():
            self.mem[name] = value.copy()

    def run(self, term):
        self.stmt(term, {})
        return self.mem

    # Statements

    def stmt(self, t, env):
        if not isinstance(t, list):
            return  # dummy
        op = t[0]
        if op == "seq":
            self.stmt(t[1], env)
            self.stmt(t[2], env)
        elif op in ("loop", "ploop", "sloop"):
            start, end, tile, var, body = t[1:]
            step = tile if isinstance(tile, int) else DEFAULT_TILE
            private = self.private_tensors(body, var) if op == "ploop" else []
            for v in range(start, end, step):
                # Each program of a parallel loop has its own copy of the
                # intermediates it writes without indexing by the loop
                # variable, as in Trinity's generated kernels.
                for name in private:
                    self.mem[name][...] = 0
                self.stmt(body, {**env, var: (v, step)})
        elif op == "store":
            base, val, index = t[1:]
            names = base[1].split(",")
            for i, name in enumerate(names):
                value = self.expr(val, env, i)
                target = self.mem[name]
                target[self.slices(index, env, target.ndim)] = value
        else:
            raise ValueError(f"unknown statement {op}")

    @staticmethod
    def private_tensors(body, var):
        names = set()
        for t in walk(body):
            if isinstance(t, list) and t[0] == "store" and t[1][0] == "tensor":
                index = t[3]
                if not any(isinstance(c, list) and c[0] in ("tile", "elem") and c[1] == var
                           for c in index[1:]):
                    names.update(t[1][1].split(","))
        return sorted(names)

    def slices(self, index, env, ndim):
        comps = index[1:] if isinstance(index, list) else []
        out = []
        for c in comps:
            if c == "fulltile":
                out.append(slice(None))
            elif c[0] == "tile":
                v, step = env[c[1]]
                out.append(slice(v, v + step))
            elif c[0] == "elem":
                v, step = env[c[1]]
                out.append(slice(v // step, v // step + 1))
            elif c[0] == "const_tile":
                out.append(slice(c[1], c[1] + c[2]))
            else:
                raise ValueError(f"unknown index component {c}")
        return tuple(out + [slice(None)] * (ndim - len(out)))

    # Expressions

    def expr(self, t, env, member):
        if isinstance(t, (int, float)):
            return float(t)
        if not isinstance(t, list):
            raise ValueError(f"unexpected atom {t}")
        op, args = t[0], t[1:]
        if op == "load":
            base, index = args
            names = base[1].split(",")
            name = names[member] if len(names) > 1 else names[0]
            arr = self.mem[name]
            return arr[self.slices(index, env, arr.ndim)].copy()
        ev = lambda a: self.expr(a, env, member)
        if op in ("+", "-", "*", "/", "max", "min", "<="):
            x, y = ev(args[0]), ev(args[1])
            return {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide,
                    "max": np.maximum, "min": np.minimum,
                    "<=": lambda a, b: (a <= b).astype(float)}[op](x, y)
        if op == "exp":
            return np.exp(ev(args[0]))
        if op == "sqr":
            x = ev(args[0])
            return x * x
        if op == "sqrt":
            return np.sqrt(ev(args[0]))
        if op == "sigmoid":
            return 1 / (1 + np.exp(-ev(args[0])))
        if op == "erf":
            return np.vectorize(math.erf)(ev(args[0]))
        if op == "abs":
            return np.abs(ev(args[0]))
        if op == "cast":
            return ev(args[1])
        if op == "@":
            return np.matmul(ev(args[0]), ev(args[1]))
        if op in ("rsum", "rmax", "rmin"):
            fn = {"rsum": np.sum, "rmax": np.max, "rmin": np.min}[op]
            return fn(ev(args[0]), axis=args[1])
        if op in ("bcast", "unsqueeze"):
            return np.expand_dims(ev(args[0]), args[1])
        if op == "squeeze":
            return np.squeeze(ev(args[0]), axis=args[1])
        if op in ("permute3", "permute4"):
            return np.transpose(ev(args[0]), args[1:])
        if op == "transpose":
            return np.swapaxes(ev(args[0]), -1, -2)
        if op == "concat":
            return np.concatenate([ev(args[0]), ev(args[1])], axis=args[2])
        raise ValueError(f"unknown operator {op}")


def random_inputs(shapes, names, seed=0):
    rng = np.random.default_rng(seed)
    return {n: rng.standard_normal(shapes[n]) * 0.02 for n in sorted(names) if n in shapes}


def compare(reference, result, names):
    """Largest relative error over NAMES (infinite if anything is not finite),
    and the tensor where it occurs."""
    worst, where = 0.0, None
    for n in sorted(names):
        a, b = reference[n], result[n]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return math.inf, n
        scale = max(np.abs(a).max(), 1e-12)
        err = float(np.abs(a - b).max() / scale)
        if err > worst:
            worst, where = err, n
    return worst, where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance", nargs="?")
    ap.add_argument("--original", type=Path, help="input program, instead of the instance's")
    ap.add_argument("--shapes", type=Path, help="shapes file, instead of the instance's")
    ap.add_argument("--program", nargs="*", type=Path, help="program files to check instead")
    ap.add_argument("--tolerance", type=float, default=1e-6)
    args = ap.parse_args()
    # numpy's matmul on macOS raises spurious floating-point warnings; results
    # are checked for finiteness explicitly instead.
    np.seterr(all="ignore")

    if not args.instance and not (args.original and args.shapes and args.program):
        ap.error("give an INSTANCE, or --original, --shapes and --program")
    original = parse((args.original or ROOT / "inputs" / f"{args.instance}.ir").read_text())
    shapes = read_shapes(args.shapes or ROOT / "inputs" / f"{args.instance}.shapes")
    inputs = random_inputs(shapes, bases(original, "input"))
    checked = bases(original, "output") | (bases(original, "input") & written_names(original))
    reference = Interpreter(shapes, inputs).run(original)
    for n in sorted(checked):
        r = reference[n]
        if not np.isfinite(r).all():
            sys.exit(f"reference output {n} is not finite")
        print(f"reference {n}: shape {r.shape}, max |value| {np.abs(r).max():.3g}")

    paths = args.program or [ROOT / "results" / f"{args.instance}.ir"]
    programs = [(p.name, parse(p.read_text())) for p in paths]

    failures = 0
    for label, program in programs:
        try:
            err, where = compare(reference, Interpreter(shapes, inputs).run(program), checked)
            ok = err <= args.tolerance
            print(f"{label}: {'ok' if ok else 'MISMATCH in ' + where}"
                  f" (max relative error {err:.2e})")
        except Exception as e:  # noqa: BLE001 - report and continue
            ok = False
            print(f"{label}: ERROR {type(e).__name__}: {e}")
        failures += not ok
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
