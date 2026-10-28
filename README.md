# trinity-ggs

A port of [Trinity](../trinity)'s optimizer (tile-level equality saturation for
tensor programs, ASPLOS 2026) from Rust/egg to Common Lisp on
[GGS](../ggs) EqSat, with an analytical CPU cost model in place of Trinity's
GPU profiling.

Given a tile-level program (the same IR Trinity consumes), it saturates an
e-graph with Trinity's rewrite rules and extracts the cheapest program under
the cost model. No GPU is involved anywhere.

## Setup

Requires SBCL and Quicklisp. GGS depends on `lp-hash-table`, which is not in
Quicklisp:

```bash
git clone https://github.com/kchanqvq/lp-hash-table ~/quicklisp/local-projects/lp-hash-table
```

GGS is expected next to this repository (`../ggs`); set `GGS_DIR` otherwise.
`scripts/load.lisp` registers both and loads everything through Quicklisp.

## Usage

```bash
sbcl --dynamic-space-size 16384 --load scripts/load.lisp
```

```lisp
(in-package :trinity-ggs)
(optimize-instance "vanilla_llama")   ; program written to results/vanilla_llama.ir
(run-benchmarks)                      ; all instances, summary table
(compare-with-trinity "vanilla_llama")
```

Or non-interactively, all instances plus the comparison with Trinity:

```bash
sbcl --dynamic-space-size 16384 --non-interactive --load scripts/bench.lisp
```

Check that the optimized program computes the same thing as the input, by
interpreting both on random data with numpy:

```bash
python scripts/check_equivalence.py vanilla_llama
```

`scripts/walkthrough.lisp` prints every stage for one small program
(`examples/chain.ir`): the input and its cost, which rules fire in each
iteration, the terms extraction compares in each e-class, and the result:

```bash
sbcl --non-interactive --load scripts/walkthrough.lisp
```

Run the tests with `(asdf:test-system :trinity-ggs)`.

## Inputs

`inputs/` holds the 14 benchmark instances Trinity's optimizer tests run
(vanilla, prenorm, qknorm, keyformer, roco, gqa_vanilla and ffn attention/MLP
blocks, each for Llama-3-8B and Falcon-7B shapes): the IR, the tensor shapes,
and Trinity's iteration limit. They are extracted from
`trinity/optimizer/tests/*.rs` by `scripts/extract_inputs.py`.

`reference/trinity-winners/` holds the programs Trinity selected by GPU
profiling (from `trinity/backend/results/`).

## How it maps to Trinity

| Trinity (Rust) | Here |
| --- | --- |
| `language.rs` `TileLang` | plain s-expressions; `src/ir.lisp` reads/prints them |
| `LoopAnalysis` read/write sets, shapes | GGS analyses `acc` and `shape` (`src/analysis.lisp`) |
| `dependency.rs`, `utils.rs` guards | `src/analysis.lisp` |
| `rules.rs`, `applier.rs` | `src/rules.lisp`, one GGS rule set per rule |
| `modify` hook (seq flattening, value forwarding, `tmp_loop`) | rules `seq-assoc`, `value-forwarding`, and renaming inside `loop-fusion` |
| `run_until_saturated` + `BackoffScheduler` | `src/run.lisp`, per-rule match limits as in Trinity |
| `extract_with_cost.rs` (two passes) + GPU profiling of every candidate | `src/extract.lisp`, one cost-driven extraction with `src/cost.lisp` |
| `postprocess.rs` | `src/postprocess.lisp` |

### Deliberate differences

- **No global unions for forwarding and renaming.** Trinity's `modify` hook
  unions e-classes globally (e.g. `(tile h)` with `(elem n)` when fusing a
  loop over heads into a loop over 128-wide column tiles) and repairs the
  extracted terms afterwards. GGS has no modify hook, so these are appliers
  that build a rewritten copy of just the affected sub-e-graph.
- **Renaming respects scoping.** The renaming never touches loops that rebind
  the variable, and a fusion is skipped if the kept loop's body already refers
  to the dropped variable.
- **Extraction is a single cost-driven pass.** Trinity enumerates candidates
  (bounded by sequential loops and kernels), completes each with a FLOPs-based
  extractor, and profiles them all on a GPU. Here one bottom-up extraction picks
  the cheapest term under the cost model; see below.
- **Accumulations are not repeated.** Trinity moves a statement into (or out
  of) a loop whenever it does not depend on the loop variable, which changes
  how often an accumulation like `(store a (+ (load a i) x) i)` runs. That is
  sound only when the loop is parallel and `a` is an intermediate, which each
  iteration then recomputes privately (Trinity's prenorm kernels rely on
  this). Here loop insertion and deletion refuse statements that accumulate
  into inputs or outputs, and the optimizer signals an error if the extracted
  program repeats an accumulation in a sequential loop (it never has).
- Kept from Trinity: the lax index comparison in dependency checks, and its
  rule for choosing parallel and sequential loops (`decide-loop-types`).

## Extraction

`extract-best` (`src/extract.lisp`) is egg-style extraction: a bottom-up
fixpoint where each e-class keeps the cheapest term built from its children's
current best terms. The cost model is not a sum over nodes, so each candidate
term is scored as a program fragment on its own, after choosing its loop types:

- a statement with no free loop variables as a program (its top-level
  statements are kernels), one with free loop variables as a single kernel,
  since it sits inside a loop;
- a tensor expression as a kernel computing it once;
- indices and names by size only.

Intermediates count as on chip only if the fragment writes them, so reading a
tensor produced elsewhere is paid for. Terms of one class share their contexts,
so scoring them out of context ranks them sensibly, but this is a heuristic:
a fragment's cost inside a kernel is not exactly its cost alone.

Each class keeps one best term per set of free loop variables, and the root
takes the term with none. Trinity's rules equate statements with copies that
drifted out of the loop binding their index, and alone such a copy looks cheap
(an access indexed by an unbound variable counts as one tile). Keying by free
variables only ever compares terms that need the same enclosing loops.

Loop types (`ploop`/`sloop`) are not part of the e-graph, as in Trinity: they
are a function of the program (`decide-loop-types`), applied when scoring each
fragment and again in postprocessing.

## Cost model

Each top-level statement is a kernel, estimated with a roofline
(`src/cost.lisp`, constants in `*hardware*`, defaulting to an H100 SXM):

```
time = launch + max(bytes / bandwidth,
                    matmul FLOPs / tensor-core peak + other FLOPs / SIMT peak)
              / min(1, grid / #SMs)
```

- Bytes count global-memory traffic. An intermediate stays on chip if one
  kernel uses it and all its accesses walk the same sequential loops tile by
  tile; otherwise it is materialized, as in Trinity's code generator.
- Accesses repeat per parallel-loop iteration, and per sequential-loop
  iteration only when indexed by that loop.
- FLOPs follow Trinity's per-tile counts times all enclosing trip counts, so
  recomputation is charged.
- Symbolic tile sizes are chosen per kernel from {16, 32, 64, 128}, standing
  in for autotuning.

## Verification

`scripts/check_equivalence.py` interprets the IR with numpy (loops run
sequentially, symbolic tiles of size 64, intermediates written inside a
parallel loop without indexing by its variable are private to each
iteration, as in Trinity's generated kernels). It runs the input program and
the optimized one on the same random inputs and compares every output and
every input the program writes (the KV cache). It treats every loop as
sequential, so it cannot catch a loop wrongly marked parallel.

Every program Trinity picked by GPU profiling (`reference/trinity-winners/`)
passes this check, which calibrates the interpreter, and so does our program
for every instance. (An earlier version of this port, which enumerated
candidates like Trinity, also passed on all 1618 of them once the accumulation
guard was in place; before it, 166 failed, mostly in FFN.)

## Results

All 12 instances, run with Trinity's own iteration limits (`scripts/bench.lisp`,
on an Apple M-series laptop). Costs are cost-model estimates in microseconds;
"Trinity best" is the cheapest of Trinity's own candidate list under the same
model, and "GPU picks" are the programs Trinity chose by profiling, also scored
by the model. Every program passes `scripts/check_equivalence.py`.

These numbers predate a fix to the scheduler's ban length (bans lasted one
iteration longer than in egg) and will be regenerated.

| instance | iters | enodes | input | ours | Trinity best | GPU picks | time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ffn_llama | 10 | 4718 | 204.5 | 179.8 | 166.1 | 184.9 | 1 |
| ffn_falcon | 10 | 4718 | 239.8 | 214.3 | 214.3 | 220.6 | 1 |
| keyformer_llama | 8 | 48639 | 294.3 | 48.2 | 79.6 | 80.9–87.3 | 78 |
| keyformer_falcon | 8 | 48639 | 210.2 | 45.6 | 51.3 | 52.5–53.8 | 83 |
| prenorm_llama | 10 | 137363 | 294.6 | 49.4 | 75.6 | 75.6 | 200 |
| prenorm_falcon | 8 | 94739 | 208.3 | 49.1 | 47.9 | 47.9 | 83 |
| qknorm_llama | 8 | 62842 | 294.0 | 44.3 | 70.5 | 42.7–74.1 | 43 |
| qknorm_falcon | 8 | 62842 | 203.3 | 42.9 | 42.1 | 42.1 | 46 |
| roco_llama | 8 | 74091 | 330.2 | 44.4 | 74.5 | 74.5–84.8 | 216 |
| roco_falcon | 8 | 74091 | 243.3 | 43.1 | 46.2 | 46.2 | 202 |
| vanilla_llama | 8 | 41221 | 283.3 | 44.3 | 70.5 | 42.7–70.5 | 26 |
| vanilla_falcon | 8 | 41221 | 196.4 | 42.9 | 42.1 | 42.1 | 47 |

Caveats:

- The comparison is under our model, not measured, and extraction optimizes
  against the model directly, so its errors matter. The model does agree with
  Trinity's profiling in one useful sense: for most instances the program
  Trinity picked on the GPU is the cheapest (or near-cheapest) of its
  candidates under the model.
- No instance saturates: all stop at Trinity's iteration limit.
- Time is split roughly 2:1 between saturation and extraction overall;
  extraction takes 13–50 s on the larger e-graphs.
- FFN improves least. Its loops all have symbolic tile sizes and differently
  named variables, so only same-variable fusions apply. Trinity's best FFN
  candidate (166.1) comes from a different run configuration
  (`cost6_kern5_wo_scheduler2`).
- Trinity's `gqa_vanilla` tests feed the vanilla IR with GQA shapes, which
  disagree with its loop bounds, so they are left out.
