---
name: mutation-testing
description: Find and close test gaps in one scope of the Vortex Rust code with cargo-mutants, and refine its tests. Run the mutants, triage every survivor, add tests that kill the real gaps, propose removing tests that add nothing, ask a person about every decision that is not the agent's, and prove the result. Use when asked to mutation-test or refine the tests of a crate, file, function or diff, or when the user mentions "/mutation-testing".
---

# Mutation Testing

cargo-mutants makes small changes to the code, such as flipping `<` to `<=` or replacing a
function body, and runs the tests. A mutant the tests do not catch is a *survivor*: either the
tests never check that behaviour (a gap), or the change cannot affect any result (an equivalent
mutant). This skill turns the survivors of one scope into tests and exclusions, with evidence,
and uses the same evidence to find tests that add nothing.

It produces two things: **changes** the evidence proves, and **questions** for a person about
everything it does not. The agent finds and argues; a person decides what the code should do and
which tests may go. The report is the pull request body, and the questions are its checklist.

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

- One commit on the current branch, containing only new, extended or approved-for-removal tests
  and `.cargo/mutants.toml` entries. Do not push, open pull requests or create branches unless
  the invoking prompt says to.
- `<run>/report.json` and `<run>/report.md`, written by `$M report`. `report.md` is the pull
  request body: the changes, with their evidence, then the open questions. The caller reads
  `report.json`. Its schema is in [Reading the report](#reading-the-report).
- When a person answers questions, the next invocation applies the answers and reports again
  ([Refine: the loop](#the-loop)).

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
5. **Never weaken or loosen an existing test, and delete one only after a person approves it.**
   A test that catches no mutant is usually not useless; see [Refine](#refine-removing-tests).
6. **Equivalence needs an argument, not a hunch.** Mark a mutant `equivalent` only when you can
   say why no input can observe the change. If you cannot, it is a gap or `deferred`.
7. **Ask instead of deciding.** What the code *should* do is a person's decision. When the only
   expected value available is the current output, when current behaviour looks wrong or
   arbitrary, or when a test would pin a performance or representation choice (an array stays
   RLE after slicing, a slice keeps only the values it needs), do not write the test: record the
   survivor as `deferred` or `bug` and ask a question that gives each reading and what it leads
   to. `report` refuses a `deferred` or `bug` verdict without a question. Before you call a change
   performance- or representation-only, check that the changed representation still passes the
   constructor's `validate` and the decoder's bounds on a reachable input. For RLE these include
   `indices.len() >= values.len()` and at most 1024 values per chunk. If any input breaks them,
   the survivor is a gap: test it against the reference.

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

When the original guard sends an input to only one path, try the degenerate inputs on the other
path: empty arrays, zero chunks, all-null indices, a group longer than a chunk. A `/` or `==`
mutant of a size ratio often sends an empty array down a path that indexes `num_chunks + 1`
slots. For example, under `indices.len() / FULL_ARRAY_DECODE_RATIO` in Delta's `take`, three null
indices on an empty array reach `take_grouped` and panic.

If two paths agree only for inputs that meet a precondition, such as in-bounds indices, or "errors
now" versus "errors when executed", say so in the argument. Then treat every such survivor in the
scope the same way: either exclude all of them, naming the precondition in each comment, or defer
all of them under one question about the contract.

An equivalence that rests on an invariant needs the check that enforces it, with its file and
line, and that check must cover every way to build the value: `validate`, the Safety section of
`new_unchecked`, and deserialization. If only the checked constructor enforces it, record
`deferred` with an `equivalence` question that offers moving the check into `validate`. Before
you call a survivor unused or display-only, find its callers. If no caller can reach the mutated
code, for example an empty `0..nbuffers()` loop or a method shadowed by a trait method, it is
equivalent. If the code is public, say who outside the crate can call it.

Mutants only change code that exists, so they cannot find a missing check. For each function in
scope, list the inputs it never looks at, such as validity, offsets or empty input, and run them
against the reference. A wrong answer is a `bug` with a question, not a test to commit: the bit-
packed `is_constant` kernel ignored validity, and missed a 1024-row array, and neither showed up
as a survivor. Also run `git log --all --oneline -- <file>`: a fix on another branch that is not
on the base branch is a known bug to record.

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
  cover a single partial chunk (fewer than 1024 rows), exactly one full chunk (1024 rows at
  offset 0, or a slice from one chunk boundary to the next), a partial first chunk (a slice with
  an offset), several full chunks, a trailer, and a one-row trailer. Put patches in each, and add
  a nullable variant whose null slots hold the same value as the valid ones, so that a kernel
  that ignores validity gives a wrong answer. Use many distinct values as well as few: with few
  values, RLE and dictionary bounds never bind. Name each `rstest` case for the part it covers.
  `bitpacking/array/unpack_iter.rs` is skipped by cargo-mutants, so no mutant checks how a kernel
  walks these shapes; only these cases do.
- **Slice with the slice kernel when patches are involved.** `ArrayRef::slice` stays a lazy
  `SliceArray` over a patched `BitPacked` array. Use `<BitPacked as SliceKernel>::slice` to get a
  sliced `BitPacked` array with sliced patches.
- **Compare with a reference.** Use `assert_arrays_eq!` for arrays, with the result under test
  on the left. For scalars, compare against the same operation on the canonical source. When a
  reference exists, compare the kernel's answer with it, not with a literal you wrote. Record
  where the expected values come from as the entry's `oracle`; `report` requires it.
- **Reach the guard you mean to test.** For invalid input, assert the specific error, as
  `test_deserialize_rejects_unknown_ptype` in `encodings/fastlanes/src/rle/vtable/mod.rs` does, so
  the test cannot pass on an earlier, unrelated failure.
- **Pin the path, not only the result.** When a tuned constant such as a ratio or a threshold
  picks the path under test, size the input from the constant, or assert the precondition, in a
  test module that can see the constant. Otherwise retuning the constant silently moves the test
  to the other path, where it still passes and catches nothing.
- **Keep only cases that earn their place.** After `verify`, `$M kills <run>/verify` shows which
  cases catch which mutants. Drop a case whose kills another case also makes only if it covers no
  shape, type or nullability of its own. The kill matrix cannot see code outside the scope or
  under `mutants::skip`, so keep one case for each shape in the list above even when it catches
  nothing new, and name those cases in the report. Say what each case covers in its name or
  comment, never "these cases catch every mutant": that records a tool result that goes stale.
- Put tests in the module's `tests` module. Promote `foo.rs` to `foo/mod.rs` plus `foo/tests.rs`
  if it grows large.

Run the new tests before verifying: `cargo test -p <crate> --lib <module path>`.

### 6. Record equivalent mutants

Add one `exclude_re` entry to `.cargo/mutants.toml` per equivalent mutant or per tight family of
them, with a comment that gives the argument. Patterns match `path:line:col: description`:
escape the path's dots, match the position with `\d+:\d+`, and name the operator and function so
the entry cannot hide anything else. For a replaced function body, name the replacement values
too, as in `with [01]$`. Check that each new pattern matches only the mutants you mean:

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

Each `gap-fixed` entry also has `"oracle"`: where its tests' expected values come from, such as
`"the same slice of the source PrimitiveArray"`.

Copy mutant names exactly from `$M survivors <run> --json`. With no survivors, write
`{"entries": []}`. The verdicts are:

- `gap-fixed`: a test now kills it. Name the tests and the oracle.
- `equivalent`: an exclusion now covers it, with the argument in both places.
- `bug`: the survivor shows wrong behaviour. Describe the input and the wrong result, and ask.
- `deferred`: a real gap left for a person's decision, for example "display text only" or "pins a
  performance choice". Ask.

The same file holds `removals` and `questions`; see [Refine](#refine-removing-tests) and
[Questions](#questions).

Then:

```bash
$M report target/mutants/rle
```

`report` refuses a triage that misses a survivor, lists one twice, calls a mutant `gap-fixed`
when verify did not see it caught or without an oracle, calls one `equivalent` without a
matching exclusion, or leaves a `deferred` or `bug` verdict without a question. It also checks
every removal against the kill matrix. Fix what it lists and rerun it. On success it writes
`report.json` and `report.md`.

### 9. Get an independent review

Before you write the pull request, have a second agent review the change without your
conclusions: give it the test diff, the code under test and `triage.json`, and ask it to check
each oracle, each verdict and each question against the code, and to look for inputs no case
covers. Work read-only and never in parallel with `run`, `verify` or `probe`. Check every claim
it makes before you act on it: run the input it names against the reference, or rerun the
mutants it says are caught. On the first three scopes this found two kernel bugs and four wrong
verdicts that mutation testing alone did not.

### 10. Check and commit

Run the checks for the crate you changed:

```bash
cargo +<NIGHTLY_TOOLCHAIN from .github/workflows/ci.yml> fmt -p <crate>
cargo clippy -p <crate> --all-targets --all-features -- -D warnings
cargo test -p <crate> --lib <module path>
```

Commit the tests and exclusions together, following the Commits section of `AGENTS.md`
(sign-off as the user, never as an agent). Start the message with a summary line naming the scope
and summarize `report.md` below it; the full report, with its questions, is the pull request
body. If there are no survivors, no removals and no questions, make no commit.

## Refine: removing tests

Every `run` records a kill matrix, `<run>/kills.json`: which tests fail under each caught
mutant. `$M kills` reads it:

```bash
$M kills target/mutants/rle                         # this run
$M kills <ci-shard-dirs...> --out target/mutants/fastlanes-kills.json   # package-wide, merged
```

It lists the tests that catch no mutant, groups of tests that catch exactly the same mutants,
and a small set that catches all of them. **None of these is a removal list.** A test can catch
no mutant because the tool cannot see what it checks. Decide why before proposing anything:

| Why it catches nothing, or the same as another | Action |
|---|---|
| It checks a guard inside a macro: `vortex_ensure!`, `vortex_bail!`, `assert!`. cargo-mutants does not mutate macro arguments, so `*_rejects_*` and `should_panic` tests catch nothing | Keep |
| It is a golden or round-trip test of generated code, such as protobuf metadata | Keep |
| It is an edge case: empty, single row, boundary | Keep |
| Its cases differ in type, nullability or shape. Generic code is mutated once for every type, so mutants cannot tell `u8` from `u64` | Keep |
| It asserts only what it constructed, and another test checks the same and more | Propose (`no-kills`) |
| It has the same inputs as another test or case | Propose (`duplicate`) |

Evidence must cover the whole package: a test that catches nothing in this scope may catch
mutants elsewhere. Pass `report --kills` a matrix from a package-wide run or from the CI shards.
Logs from runs without `--no-fail-fast` miss kills by integration tests and doctests; `$M`
passes it, CI before it did not.

Record each proposal in `triage.json`, with a removal question that refs it:

```json
"removals": [
  {
    "tests": ["<module>::tests::<test>"],
    "kind": "no-kills",
    "applied": false,
    "reason": "It asserts only what it constructed; <kept test> builds the same shape and asserts all of it and more."
  }
]
```

Name a kept test that asserts everything the removed one does, with the same inputs; check it
line by line, not by its name. For `duplicate`, add `"duplicate_of"`: a kept test whose kills
include this one's. Leave
`"applied": false` until a person answers the question `remove`; then delete the test, set it to
`true`, rerun the scope into a new directory and compare:

```bash
$M run ... --out target/mutants/rle-after
$M compare target/mutants/rle target/mutants/rle-after   # exits 2 if a caught mutant now survives
```

`report` refuses an applied removal without a clean `compare`, a removal answered `keep` but
applied, and one answered `remove` but not applied.

## Questions

A question is a decision the agent must not make alone. Ask about every proposed removal, every
`deferred` or `bug` verdict, and any test that would pin a choice rather than a contract:

```json
"questions": [
  {
    "id": "Q2",
    "topic": "semantics",
    "question": "Should Delta's take reject an out-of-bounds index when called, or when the result executes?",
    "context": "Small takes fail in `check_indices` (delta/compute/take.rs:82); large ones decode fully and return a lazy take that fails on execution. Two mutants of the size ratio (take.rs:44) change only which happens. If eager is the contract, I'll add a test for a large take; if not, I'll exclude them.",
    "options": ["eager: add a test", "either: exclude as equivalent"],
    "recommendation": "eager: add a test",
    "default": "either: exclude as equivalent",
    "refs": ["<mutant or test names, copied exactly>"]
  }
]
```

`topic` is one of `removal`, `semantics`, `equivalence`, `bug` or `other`. `default` is what
happens with no answer, and must be the conservative option: keep the test, leave the code,
write no test, and add no exclusion. An exclusion hides survivors for good, so only an answer may
add one; the example above is wrong on that count, so in a real question make the default "leave
deferred". Give the context a reviewer needs to answer without opening the code, and say what
each option leads to. Every factual claim in `context` must come from a line you read: quote it,
or cite the file and line.

### The loop

1. `run`, `survivors`, `kills`: gather the evidence.
2. Triage; write tests; add exclusions; propose removals; write questions.
3. `verify`; trim cases; `report`. Open the pull request with `report.md` as its body.
4. A person answers in the pull request. Record the answers in `<run>/answers.json`:
   `{"Q1": {"answer": "remove", "by": "<reviewer>"}}`.
5. Apply them: delete approved tests, write the tests that answers call for, add exclusions.
   Rerun, `compare`, and `report` again. Repeat until no question that blocks a change is open.

## Maintaining this skill

Examples here must name code that exists on the skill's own branch, and claims that a reviewer
checked. Never copy them from a run that is still being triaged.

## Reading the report

`report.json` (`schema: 1`) has:

- `package`, `selection`: the scope, as cargo-mutants arguments.
- `run_commit`, `run_dirty`: the commit the run tested, and whether it had uncommitted changes.
- `before`: counts from the run (`caught`, `missed`, `timeout`, `unviable`).
- `verdicts`: the number of survivors under each verdict.
- `after`: counts from the rerun, when `compare` ran.
- `entries`: the triage entries, each with `status`, which maps every mutant to its verified
  result (`caught`, `timeout`, `missed`, `excluded`, `unviable`, `not-found`).
- `removals`: proposed and applied removals.
- `questions`: each question with its `answer` (or `null`) and `answered_by`;
  `open_questions` counts the unanswered.
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
