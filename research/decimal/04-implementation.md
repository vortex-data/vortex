<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 4. Implementation plan

This part splits the `Narrow` encoding (issue
[#10733](https://github.com/vortex-data/vortex/issues/10733)) and the canonical decimal fix (issue
[#10701](https://github.com/vortex-data/vortex/issues/10701)) into PRs. Each PR has a brief in
[`agents/briefs/`](agents/briefs/), written so that a coding agent can implement the PR from the
brief alone. The briefs follow [`decisions.md`](../decisions.md), including D25 to D35 from the
planning review on 2026-10-23.

## How to use the briefs

- Give a coding agent one brief, `decisions.md`, and the repository rules in `AGENTS.md` and
  `STYLE.md`. The brief names the branch, the files, the steps, the tests and the acceptance checks.
- The briefs cite `develop` at `b70206818f`. Line numbers move as PRs land, so the agent reads the
  cited code before it changes it. Code from the local integer `Narrow` rewrite is cited with its
  commit, for example `f7408481ce:vortex-array/src/arrays/narrow/mod.rs`.
- The acceptance checks are commands for the maintainer to run. An agent does not run them unless
  the maintainer asks for it.
- Every commit carries `Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>` and no AI co-author
  line.
- WR3 needs a design pass before an agent starts it. Its brief lists the open questions.

| Brief | PRs |
| --- | --- |
| [`01-fixes.md`](agents/briefs/01-fixes.md) | FX1 to FX3, independent fixes |
| [`02-integer-narrow.md`](agents/briefs/02-integer-narrow.md) | NR1 to NR6, the integer `Narrow` |
| [`03-canonical-decimal.md`](agents/briefs/03-canonical-decimal.md) | CD1 to CD8c, the canonical decimal fix and decimal `Narrow` |
| [`04-wire.md`](agents/briefs/04-wire.md) | WR1 to WR3, the wire change, and LT1 to LT3 |

## The PRs

| ID | PR | Hard dependencies | Size |
| --- | --- | --- | --- |
| FX1 | Strict decoding of stored `DecimalType` and `PType` values, in three commits | None | Medium |
| FX2 | DecimalByteParts nullability cast follows the cast-rule contract | None | Small |
| FX3 | Dictionary `fill_null` no longer truncates the fill code | None | Small |
| NR1 | In-memory integer `Narrow`, with `encode`, the deprecated `encode_values`, the serialization workaround and a flatten rule | None | Large |
| NR2 | Benchmarks for compute on integer `Narrow` | NR1 | Medium |
| NR3 | Aggregates run on the child | NR2 | Small |
| NR4 | Arithmetic, `between`, `zip` and casts run on the child | NR3 | Large |
| NR5 | Structural encodings move above `Narrow`, and dictionary codes use the child | NR4, FX3 | Medium |
| NR6 | The existing narrowing call sites move to `encode_values` | NR1 | Medium |
| CD1 | Decimal benchmarks | None | Medium |
| CD2 | Compat fixtures for old decimal forms | None | Medium |
| CD3 | One width function, imports that check and copy, and the shared helpers | None | Medium |
| CD4 | Decimal `Narrow` in memory | NR1, NR3, NR4, CD1, CD3 | Large |
| CD5 | DecimalByteParts decodes to the canonical width | CD1, CD3, released CD2 fixtures | Medium |
| CD6 | The `vortex.decimal` read plugin converts old nodes | FX1, CD3, CD4, released CD2 fixtures | Medium |
| CD7 | One DecimalByteParts scheme on the canonical array | NR1 | Medium |
| CD8a | Every decimal array has the canonical width | CD3, CD5, CD6, CD7 | Large |
| CD8b | `DecimalArray::try_new` checks every valid value, with an `ExecutionCtx` | CD8a | Large |
| CD8c | Remove the code for mixed decimal widths | CD8a | Medium |
| WR1 | `vortex.narrow` wire format for integers and decimals | NR1, CD3, CD4, after CD8a | Medium |
| WR2 | One preview edition with `vortex.narrow` and DecimalByteParts v2 | WR1, CD5, CD7 | Small |
| WR3 | A `Narrow` scheme in the compressor (needs a design pass) | WR1, WR2, NR1, NR5, CD3, CD4, CD7 | Large |

Later work: LT1 puts both formats into `core` and decides the end of the serialization
workaround. LT2 is the max pass in DecimalByteParts decoding (D16). LT3 is the pushdown of sum,
min and max into the MSP (D19). LT2 and LT3 wait until benchmarks show that they are needed.

## Graph

Each arrow is a hard dependency. The graph leaves out dependencies that follow from other arrows.
The dashed arrows are the release gate of the compat fixtures (D31).

```mermaid
flowchart TD
    subgraph FIX["Independent fixes"]
        FX1["FX1 Strict decoding"]
        FX2["FX2 DBP nullability cast"]
        FX3["FX3 Dict fill code"]
    end
    subgraph NR["Integer Narrow"]
        NR1["NR1 Encoding"]
        NR2["NR2 Benchmarks"]
        NR3["NR3 Aggregates"]
        NR4["NR4 Arithmetic and casts"]
        NR5["NR5 Structural rules"]
        NR6["NR6 Call sites"]
    end
    subgraph CD["Canonical decimal"]
        CD1["CD1 Benchmarks"]
        CD2["CD2 Compat fixtures"]
        REL(["Release with CD2"])
        CD3["CD3 Width and imports"]
        CD4["CD4 Decimal Narrow"]
        CD5["CD5 DBP decoding"]
        CD6["CD6 Read plugin"]
        CD7["CD7 DBP scheme"]
        CD8a["CD8a Canonical width"]
        CD8b["CD8b Value check"]
        CD8c["CD8c Dead code"]
    end
    subgraph WR["Wire change"]
        WR1["WR1 Wire format"]
        WR2["WR2 Preview edition"]
        WR3["WR3 Narrow scheme"]
    end

    NR1 --> NR2 --> NR3 --> NR4 --> NR5
    FX3 --> NR5
    NR1 --> NR6
    CD2 --> REL
    REL -.-> CD5
    REL -.-> CD6
    CD1 --> CD4
    CD3 --> CD4
    NR1 --> CD4
    NR3 --> CD4
    NR4 --> CD4
    CD1 --> CD5
    CD3 --> CD5
    FX1 --> CD6
    CD3 --> CD6
    CD4 --> CD6
    NR1 --> CD7
    CD3 --> CD8a
    CD5 --> CD8a
    CD6 --> CD8a
    CD7 --> CD8a
    CD8a --> CD8b
    CD8a --> CD8c
    CD8a --> WR1
    CD4 --> WR1
    WR1 --> WR2
    CD5 --> WR2
    CD7 --> WR2
    WR2 --> WR3
    NR5 --> WR3
```

FX2 has no edge. It can land at any time.

## Order of work

Several agents can work at once. A wave starts when its dependencies have merged.

| Wave | PRs | Note |
| --- | --- | --- |
| 1 | FX1, FX2, FX3, NR1, CD1, CD2, CD3 | No dependencies. The next regular release after CD2 publishes the fixtures |
| 2 | NR2, NR6, CD7, CD5 | NR6 merges after NR5 (D35). CD5 waits for the release with CD2 |
| 3 | NR3, then NR4 | One stack. Merge NR1 and NR2 before reading the CodSpeed reports of NR3 and NR4 |
| 4 | NR5, CD4 | NR5 waits for FX3 |
| 5 | CD6 | Waits for FX1, CD4 and the release with CD2 |
| 6 | CD8a | The last in-memory step that the wire change needs |
| 7 | CD8b, CD8c, WR1 | Independent of each other |
| 8 | WR2, then WR3 | WR3 after its design pass. If possible, the wire stack ships in the same release as the in-memory change (D11) |

The critical path runs NR1, NR2, NR3, NR4, CD4, CD6, CD8a, WR1, WR2 and WR3. The release that
publishes the CD2 fixtures also gates CD5 and CD6, so CD2 is worth landing in wave 1.

## Rules for every PR

- No PR adds temporary code (D11). The serialization workaround of D26 is the one documented
  exception. It stays until `vortex.narrow` is in `core`.
- Benchmarks land before the changes that they cover (D19): NR2 before NR3 to NR5, and CD1 before
  CD4, CD5 and CD7.
- Every call of the deprecated `encode_values` carries `#[expect(deprecated)]` with a
  `TODO(connor)` reason (D28).
- No `Narrow` sits directly above a generic structural encoding after optimization (D25).
- A narrowing copy writes zeros in null slots and fails on a valid value outside the precision
  (D21). Reads without a copy trust the precision and never panic (D14).

## Deferred

- Which arrays the compressor narrows (D27). The WR3 design pass decides it, together with the
  precedence between the `Narrow` scheme and the DecimalByteParts scheme and the cascade level
  (D33).
- Other members of the preview edition. WR2 adds only `vortex.narrow` and
  `vortex.decimal_byte_parts.v2`.
- Arrow import options for invalid decimal values (D13). The import fails, and the error names
  arrow-rs `null_if_overflow_precision`.

## Follow-up issues

These problems came up during planning and are outside the planned PRs. Each one gets its own
issue.

1. Generic structural encodings always sit above other encodings. NR5 does this for `Narrow` only.
2. The decimal `min_max` aggregate panics on a valid value outside the precision, which a file read
   can produce.
3. The cast rules of RunEnd, ZigZag, FoR and RLE push nullability casts down without the
   `trivially_cast_nullability` guard that FX2 adds to DecimalByteParts.
4. The compat harness: no CI job runs `--mode all`, one failing fixture aborts the whole check, and
   the README describes `--git-ref` wrongly.
5. Stale versioning docs about the decimal scheme and per-edition scheme configuration.

## Risks

| Risk | Mitigation |
| --- | --- |
| CD5 changes the CUDA decoder, which few reviewers know | A minimal change in its own commit, with a CUDA reviewer. It becomes its own PR if it grows |
| NR6 writes signed internal children that earlier releases have not read | Readers on `develop` accept any integer type there. NR6 runs the latest release on its files |
| The `take_map` regressions on #10333 have no known cause | NR6 compares CodSpeed runs on NR1, NR3, NR4 and NR6, and reads the flame graph diff |
| WR3 has open design questions | A design pass before any code. The other wire PRs do not depend on it |
| The decimal critical path is long | Wave 1 starts seven PRs at once, and CD5, CD7 and NR6 need only early PRs |
