<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Handoff prompt: protobuf extension type

Copy everything below the line into a new agent session that runs in a clone of
`vortex-data/vortex` with the branch `ct/protobuf-ext` checked out.

---

You are taking over an experimental branch, `ct/protobuf-ext`, in the Vortex repository. The
branch was written by an earlier agent session for Connor Tsui (branch
prefix `ct/`). Connor will use you for two kinds of work:

1. Exploring the change: answering questions about how it works, why it was built this way, and
   what it would take to change it.
2. Splitting improvements out of the branch into isolated changes on `develop`, one PR or one
   stacked branch at a time.

Do not start either kind of work until Connor asks. First build your understanding as described
below, then report back in a short summary and wait for instructions.

## What the branch is

The branch stores many protobuf messages of one schema as a Vortex extension type,
`vortex.protobuf`, with one Vortex column per protobuf field. Query engines such as DataFusion then
read only the fields that a query uses. On the way, the work fixed several general scan gaps that
are useful without protobuf. Nothing on the branch is merged or stable.

The full reference is `research/protobuf/README.md`. Read all of it first. It has the
architecture, a table of the eight commits, every design decision, the file map of the new crate,
the format changes, measurements, known gaps, and a suggested way to split the branch.

## Build your understanding in this order

1. Read the repository rules: `AGENTS.md` (root), `STYLE.md`, `vortex-array/src/dtype/AGENTS.md`,
   and the nearest `AGENTS.md` of any crate you touch. They override your defaults. Key ones:
   commits need a DCO sign-off by Connor (`git commit -s`, author
   `Connor Tsui <connor.tsui20@gmail.com>`); PR titles and bodies follow the rules in `AGENTS.md`;
   verification commands run only when Connor asks for them.
2. Read `research/protobuf/README.md` in full.
3. Read the opaque extension type plan that this work follows: `git fetch origin
   ct/ext-types-research`, then `research/extension-types/README.md` and `03-design.md` on that
   branch. The rules that matter here: structural operations always work, semantic operations need
   claims (`Eq`, `Ord`), storage is read only through `ext_storage`, and there is no public
   `ext_wrap` expression. Connor assumes that plan will merge.
4. Walk the commits in order with `git log --reverse origin/develop..ct/protobuf-ext` and
   `git show --stat <commit>`. For each, read the commit message and the main files listed in the
   README. Note that `7ac0dbbd0` corrects `91dbd9bf9`; the message of `91dbd9bf9` describes a NaN
   approach that was later undone.
5. Read the crate docs in `vortex-protobuf/src/lib.rs`. They are the source of truth for the
   mapping, the options and the round-trip rules.
6. Read these core pieces closely, because most questions and splits touch them:
   - `vortex-layout/src/layouts/extension/` (the `vortex.ext` layout: writer, reader, plan,
     `rebind_to_storage`)
   - `vortex-layout/src/layouts/table.rs` (where the writer decides to split an extension)
   - `vortex-scan/src/plan/builtins/pack/mask.rs` (`lift_masks`, `holds_under_masking`)
   - `vortex-layout/src/layouts/zoned/writer.rs` (statistics beneath nullable structs)
   - `vortex-layout/src/layouts/list/expr/` and `reader/` (classic list element pushdown)
   - `vortex-scan/src/plan/builtins/list_pack/` (planner list support)
   - `vortex-layout/src/layouts/struct_/reader.rs` (`expand_struct_root`, the strict-function rule)
   - `vortex-datafusion/src/convert/exprs/mod.rs` (the `GetFieldFunc` branch) and
     `vortex-datafusion/src/convert/exprs/partial/` (partial projection pushdown)
   - `vortex-protobuf/src/mapping.rs`, `decode/message.rs`, `decode/builders.rs`,
     `encode/columns.rs`, `metadata.rs`, `canonical.rs`
7. Check what changed on `develop` since the branch base `f5b77122a`
   (`git log f5b77122a..origin/develop --oneline`), especially in vortex-scan, vortex-layout,
   vortex-datafusion, vortex-edition and extension types. Note conflicts that a rebase would hit.
8. Check the state of the related PRs and issues with the GitHub tools: #11431, #11432 (mixed
   DataFusion expression trees, overlapping partial pushdown), #11160 (normalize child validity
   beneath nullable structs, overlapping the zone map change), #11595 and #11534 (unstable edition
   and benchmarks), #9205 (`is_null` pruning beneath nullable structs), and the extension type
   tracking issues #11114 and #11115.

Then report: your one-paragraph understanding of the branch, anything in the README that the code
contradicts, the rebase conflicts you expect, and the state of the related PRs.

## Facts that are easy to get wrong

- Field access on a protobuf value is `get_item` over `ext_storage`, built by
  `vortex_protobuf::protobuf_field`. `get_item` directly on an extension operand fails to bind on
  purpose.
- Splitting an extension's storage into columns is opt-in per extension type, through
  `LayoutSession::decompose_extension`, and only when the enabled editions include `vortex.ext`.
  Extension types that do not opt in (temporal, spatial) keep their leaf layout and statistics.
- `vortex.list` is NOT in the unstable edition. PR #11595 removed it on purpose, and vortex-bench
  enables unstable for every suite. Tests that need list layouts include `vortex.list` with an
  explicit edition inclusion.
- `vortex.all_non_nan` is NOT in the unstable edition either. Fields beneath nullable structs record
  the NaN count, which can only overcount, so `nan_count == 0` still proves a zone has no NaN.
- The zone maps beneath nullable structs are written by every writer in every edition. That is a
  format change outside the experiment.
- Reads require the stored storage dtype to equal the declared one, so the descriptor-to-dtype
  mapping is part of the file format. Mapping version 1 must keep giving the same dtype forever. A
  change to the mapping needs a new mapping version or a new option, never an edit of an existing
  one.
- Pushed-down nested field access under a null parent returns null; DataFusion's own `get_field`
  returns the stored child. This is an existing converter choice that partial pushdown applies in
  more places.
- The prost-reflect codec in `vortex-protobuf/src/tests/oracle.rs` is a test oracle only, and it
  differs from protobuf parse rules in ways that `src/tests/semantics.rs` documents.
- Protobuf editions syntax is not supported because the upstream crates cannot parse it.

## When you split out a change

- Work on a new branch from the latest `origin/develop`, named with Connor's prefix, for example
  `ct/nullable-struct-planner-reads`. Never push to `ct/protobuf-ext` unless Connor asks.
- Use the split plan in the README as a starting point, and confirm the order with Connor. The
  general scan changes (nullable struct reads on the planner, zone maps beneath nullable structs,
  list element pushdown, planner list support, partial projection pushdown, `get_field` through
  extension storage) do not need protobuf. Their tests must use plain structs, lists or a test
  extension type, not vortex-protobuf.
- Move code by reading it and rewriting it cleanly for the target, not by blind cherry-picks. The
  branch commits mix concerns: `91dbd9bf9` holds both the planner change and the zone map change,
  `d318b6ea0` holds both the classic and the planner list work, and `0cb27a3cf` holds both the
  protobuf SQL functions and follow-up fixes to partial pushdown.
- Keep each change self-contained: its own tests, docs and edition changes, and nothing it does not
  need. Check every new line against the "Common Mistakes" list in `AGENTS.md`.
- Each split change gets a commit message and, if Connor asks for a PR, a PR body that follows
  `AGENTS.md`: why the change is needed and why this approach, in plain prose; call out behavior
  and format changes; one `changelog/*` label. A change with a benchmark lands its benchmark first.
- Flag decisions that need Connor before you build them. The README's "Decisions to review" list
  has the open ones, for example whether the zone map change should wait for #11160, how partial
  pushdown relates to #11431 and #11432, and the layout ID name.
- Do not create a PR unless Connor asks.

## Building and testing

- Disk and CPU are usually tight. Use the `ci` profile, no incremental cache, and scope every
  command to crates. Never build the whole workspace.

  ```bash
  CARGO_INCREMENTAL=0 cargo test --profile ci -p <crate> [filter]
  ```

- If disk runs low, delete `target/ci/incremental`, `target/debug` and `target/release`, and use
  `cargo clean --profile ci -p <crate>` for crates you no longer need.
- Format with the nightly pinned as `NIGHTLY_TOOLCHAIN` in `.github/workflows/ci.yml`. Run clippy
  with `-D warnings`. Edition records under `vortex/editions/` are generated with
  `cargo run -p xtask -- generate-editions` and checked with `check-editions`.
- Run checks only when Connor asks for them, and report what you ran and what you could not run.

## Writing style

Connor wants plain, simple English in prose: short sentences, no em dashes, no filler. Answer
first, then explain. Use tables and lists only when the content has that shape. Code comments
explain only non-obvious logic.
