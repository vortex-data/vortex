---
name: mutation-testing
description: Find and close test gaps in one scope of the Vortex Rust code with cargo-mutants. Run the mutants, triage every survivor, add tests that kill the real gaps, record equivalent mutants, and prove the result. Use when asked to mutation-test a crate, file, function or diff, or when the user mentions "/mutation-testing".
---

# Mutation Testing

cargo-mutants makes small changes to the code, such as flipping `<` to `<=` or replacing a
function body, and runs the tests. A mutant the tests do not catch is a *survivor*: either the
tests never check that behaviour (a gap), or the change cannot affect any result (an equivalent
mutant). This skill turns the survivors of one scope into tests and exclusions, with evidence.

Work through the steps in order. `scripts/mutants.py` does the mechanical parts and checks your
work; never run cargo-mutants by hand for steps it covers.

```bash
M=.agents/skills/mutation-testing/scripts/mutants.py
```

## Contract

**Input: one scope.** A crate (`--package`), narrowed by any of: file globs (`--file`), a mutant
name regex (`--re`), a diff (`--in-diff`), and cargo-mutants selection arguments after `--`
(such as `--shard 2/8`). Whoever invokes the skill chooses the scope; sharding and incremental
selection by commit belong to that caller. Never widen the scope or test other scopes.

**Output.**

- One commit on the current branch, containing only new or extended tests and
  `.cargo/mutants.toml` entries. Do not push, open pull requests or create branches unless the
  invoking prompt says to.
- `<run>/report.json` and `<run>/report.md`, written by `$M report`. The caller reads
  `report.json`. Its schema is in [Reading the report](#reading-the-report).

When the scope has no mutants or no survivors, still write the report and make no commit.

## Rules

1. **Do not touch the tree while `run`, `verify` or `probe` runs.** They build in place, and
   cargo-mutants and the probe write into source files. Each command checks on exit that it
   restored the tree, and stops loudly if not. If that happens, stop and report: do not try to
   repair the tree from memory.
2. **Add tests; do not change production code.** The only exceptions are an exclusion in
   `.cargo/mutants.toml` and the `#[cfg_attr(mutants, mutants::skip)]` parse workaround in
   [Troubleshooting](#troubleshooting). If a survivor exposes a real bug, record it with the
   verdict `bug` and leave the code alone.
3. **Expected values come from the contract or a reference implementation, never from the
   current output.** Compare against the same operation on canonical data, such as a
   `PrimitiveArray`, or state values that follow from the input. Copying what the code returns
   today hides bugs instead of finding them.
4. **Every new test must kill a recorded survivor.** `verify` proves this. A test that kills
   nothing does not belong in this commit.
5. **Never weaken, delete or loosen an existing test.**
6. **Equivalence needs an argument, not a hunch.** Mark a mutant `equivalent` only when you can
   say why no input can observe the change. If you cannot, it is a gap or `deferred`.

## Steps

### 1. Preflight

```bash
cargo mutants --version   # CI pins the version in $M (CARGO_MUTANTS_VERSION)
git status --short
```

Install a missing cargo-mutants with
`cargo install --locked cargo-mutants@<CARGO_MUTANTS_VERSION from $M>`. The tree may carry
uncommitted work: runs record it, and it becomes part of the baseline.

### 2. Run the scope

```bash
$M run --package vortex-fastlanes --file 'encodings/fastlanes/src/rle/**' \
  --out target/mutants/rle
```

Name each `--out` directory after the scope; it must be new and git-ignored, so use `target/`.
The command lists the scope first and refuses more than 500 mutants (`--max-mutants`), because
the caller owns sharding. Expect about 4 to 8 seconds per mutant: builds are incremental, and
`$M` turns sccache off because sccache disables incremental compilation.

Exit status 4 from cargo-mutants means the tests fail before any mutation. `run` stops and points
to the baseline log. Report that failure; nothing can be judged until it is fixed.

### 3. Read the survivors

```bash
$M survivors target/mutants/rle
```

Survivors are grouped by function, most suspicious first, with the source and the mutated lines
marked. Timed-out mutants count as detected and need no triage.

### 4. Triage every survivor

Look at each group as a whole before its mutants. Several survivors in one function, or a
function whose whole body can be replaced (`[whole body replaceable]`), usually share a single
cause: no test reaches the function, or no test checks its result.

| Signal | Likely verdict | Next step |
|---|---|---|
| Whole body replaceable, or most of a function's mutants survive | Gap: unreached or unchecked | Probe, then test the function directly |
| A boundary operator survives (`<` → `<=`, `+` → `-` in an index) | Gap: an edge is untested | Test the boundary: chunk edges, slice offsets, one-element tails |
| The change only picks between two paths that give the same result (thresholds, sparse versus dense, SIMD versus scalar) | Equivalent, but check both paths' limits | Exclude it, but only if *both* paths are valid for *every* input the original routes to either |
| The guarded operation is a no-op at the boundary (`fill` of an empty slice, a loop of zero iterations) | Equivalent | Exclude it |
| `Display`, `Debug` or error-message text | Gap of low value | `deferred`, unless the text is a contract |
| The surviving behaviour is wrong, so the tests pass only because they never look | Bug | `bug`; do not fix |

Check the equivalent row carefully. One path that is only valid for some inputs makes the mutant
detectable. For example, in `take_grouped` a group can hold more positions than the selective
decode buffer, so forcing selective decoding can panic. Those mutants are gaps, not equivalents.

To tell "unreached" from "reached but unchecked", probe a line that starts a statement in the
function:

```bash
$M probe --package vortex-fastlanes --file encodings/fastlanes/src/rle/kernel.rs --line 57 --lib
```

The probe inserts a temporary `panic!`, runs the tests, restores the file, and lists the tests
that reached the line. `NOT REACHED` means the existing tests never execute the line, whatever
they assert.

### 5. Write the tests

Follow `AGENTS.md` and `STYLE.md`. Then, for this kind of test:

- **Call the kernel directly when the public entry point can answer without it.** Entry points
  such as `is_constant` answer from cached or exact statistics before dispatching, so a test
  through them may never reach the kernel. Call the vtable or kernel instead, for example
  `BitPackedIsConstantKernel.aggregate(...)` or `<BitPacked as CompareKernel>::compare(...)`.
  `encodings/fastlanes/src/bitpacking/compute/is_constant.rs` and
  `encodings/fastlanes/src/delta/compute/is_constant.rs` show the pattern. Also check the public
  answer against the reference, so the test covers both.
- **Build the shapes the kernel handles separately.** FastLanes code works in 1024-value chunks:
  cover a partial first chunk (a slice with an offset), full chunks, a trailer, and a one-row
  trailer, with patches or nulls in each. Name each `rstest` case for the part it covers.
- **Slice with the slice kernel when patches are involved.** `ArrayRef::slice` stays a lazy
  `SliceArray` over a patched `BitPacked` array. Use `<BitPacked as SliceKernel>::slice` to get a
  sliced `BitPacked` array with sliced patches.
- **Compare with a reference.** Use `assert_arrays_eq!` for arrays, with the result under test
  on the left. For scalars, compare against the same operation on the canonical source.
- Put tests in the module's `tests` module. Promote `foo.rs` to `foo/mod.rs` plus `foo/tests.rs`
  if it grows large.

Run the new tests before verifying: `cargo test -p <crate> --lib <module path>`.

### 6. Record equivalent mutants

Add one `exclude_re` entry to `.cargo/mutants.toml` per equivalent mutant or per tight family of
them, with a comment that gives the argument. Patterns match `path:line:col: description`:
escape the path's dots, match the position with `\d+:\d+`, and name the operator and function so
the entry cannot hide anything else. Check that each new pattern matches only the mutants you
mean:

```bash
cargo mutants -p <crate> --list | wc -l   # should drop by exactly the number you excluded
```

### 7. Verify

```bash
$M verify target/mutants/rle
```

`verify` re-tests every survivor against the current tree and records whether it is now caught,
still missed, or excluded. A `not-found` result means the mutated line changed. Since this skill
does not change production code, that should not happen; investigate if it does.

### 8. Write the triage and the report

Write `<run>/triage.json`, with one entry per cause, covering every survivor exactly once:

```json
{
  "entries": [
    {
      "verdict": "gap-fixed",
      "reason": "No test reached the patch path: `is_constant` answered from min/max stats.",
      "tests": ["bitpacking::compute::is_constant::tests::kernel_applies_patches"],
      "mutants": ["encodings/fastlanes/src/bitpacking/compute/is_constant.rs:188:18: replace - with / in apply_patches_idx_typed"]
    },
    {
      "verdict": "equivalent",
      "reason": "`chunk[..0].fill(..)` is a no-op, so `> 0` and `>= 0` agree.",
      "mutants": ["encodings/fastlanes/src/lib.rs:196:82: replace > with >= in backfill_leading_nulls"]
    }
  ]
}
```

Copy mutant names exactly from `$M survivors <run> --json`. With no survivors, write
`{"entries": []}`. The verdicts are:

- `gap-fixed`: a test now kills it. Name the tests.
- `equivalent`: an exclusion now covers it, with the argument in both places.
- `bug`: the survivor shows wrong behaviour. Describe the input and the wrong result.
- `deferred`: a real gap left for later. Say why, for example "display text only" or "needs a
  fixture this scope does not have".

Then:

```bash
$M report target/mutants/rle
```

`report` refuses a triage that misses a survivor, lists one twice, calls a mutant `gap-fixed`
when verify did not see it caught, or calls one `equivalent` without a matching exclusion. Fix
what it lists and rerun it. On success it writes `report.json` and `report.md`.

### 9. Check and commit

Run the checks for the crate you changed:

```bash
cargo +<NIGHTLY_TOOLCHAIN from .github/workflows/ci.yml> fmt -p <crate>
cargo clippy -p <crate> --all-targets --all-features -- -D warnings
cargo test -p <crate> --lib <module path>
```

Commit the tests and exclusions together, following the Commits section of `AGENTS.md`
(sign-off as the user, never as an agent). Start the message with a summary line naming the scope
and paste `report.md` below it. If `report` found no survivors, make no commit.

## Reading the report

`report.json` (`schema: 1`) has:

- `package`, `selection`: the scope, as cargo-mutants arguments.
- `run_commit`, `run_dirty`: the commit the run tested, and whether it had uncommitted changes.
- `before`: counts from the run (`caught`, `missed`, `timeout`, `unviable`).
- `verdicts`: the number of survivors under each verdict.
- `entries`: the triage entries, each with `status`, which maps every mutant to its verified
  result (`caught`, `timeout`, `missed`, `excluded`, `unviable`, `not-found`).
- `timeouts`: mutants that timed out in the run.
- `files_changed`: files that differ from `run_commit`.

## Troubleshooting

- **`failed to parse <file>`**: cargo-mutants parses every module with `syn`, which rejects some
  macro input, such as the `#[gat]` associated types in `bitpacking/array/unpack_iter.rs`. Put
  `#[cfg_attr(mutants, mutants::skip)]` on the `mod` item that loads the file. `cfg(mutants)` is
  declared in the workspace `check-cfg` list and never set, so rustc ignores the attribute. This
  skips the whole module, so say so in the report.
- **Every mutant times out, or builds are slow**: check that nothing re-enables sccache. `$M`
  clears `RUSTC_WRAPPER` and sets `CARGO_INCREMENTAL=1` for its own commands only.
- **A command was interrupted**: cargo-mutants restores the file it was mutating when
  interrupted with Ctrl-C. Before continuing, check `git status` and confirm that only your
  intended edits remain.
- **The scope is larger than `--max-mutants`**: ask the caller for a narrower scope instead of
  raising the limit.
