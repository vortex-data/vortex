<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Decisions

This file records the decisions that the maintainer made while reviewing these notes on
2026-10-19, 2026-10-22 and 2026-10-23. It also lists the questions that stay open. The other
documents follow this file. If a document disagrees with it, this file wins.

"Settled" marks a choice of the maintainer. "Recommendation" marks a proposal of these notes that
the maintainer has not confirmed yet.

## Extension types

### Settled

| ID | Decision | Reason |
| --- | --- | --- |
| X1 | Extension types are opaque by default. No code applies storage semantics to an extension value unless the type opts in | Storage semantics give wrong results for most types (EXT-Q2, EXT-C2) |
| X2 | A derive (`eq` or `ord`) is a property of one extension dtype. It never applies across dtypes | Two dtypes with different metadata are different types |
| X3 | Engines convert operands to a common dtype. Vortex runs only operations whose operand dtypes already match | Engines already do this in their planners, and one owner avoids two rule sets that disagree |
| X4 | A type implements its behavior as rewrite rules on arrays. The rules are registered with the type, together with its compression scheme and the other parts of the type. There is no `ext_wrap` expression | A rule builds its result array directly, so a separate wrap expression adds nothing |
| X5 | A rewrite rule pulls generic structural arrays, such as `Constant` and `Dict`, out from under the extension wrapper | Code that looks for a constant or a dictionary then finds it at the top, without any knowledge of extension types |
| X6 | Extension values inside a struct, list or map are handled by removing extension layers at every depth. Nested kernels get no extension code | Kernels then see storage dtypes only |
| X7 | `ExtensionArray::try_new` checks every non-null value. A new `ExtensionArray::new_unchecked` skips the check for kernels that keep legal values | Today arrays are never checked (EXT-V1). This is a bug |
| X8 | API changes are acceptable where correctness needs them, for example fallible constructors and private enum variants | Correctness and clear abstractions matter more than the current API |
| X9 | Whether a validity query raises the errors of values that it skips is out of scope here | Issue [#10313](https://github.com/vortex-data/vortex/issues/10313) owns this question, and it is not specific to extension types |

### Open

| ID | Question | Recommendation |
| --- | --- | --- |
| XQ1 | Which arrays does the rule of X5 pull out, and what stays under the wrapper? | Pull out the persistent structural encodings: `Constant`, `Dict`, `RunEnd`, `Sparse`, `Chunked`, `Masked` and `Interleave`. Slices, filters and scalar functions stay under the wrapper. A canonical extension array is a wrapper over canonical storage |
| XQ2 | Where do the rules of X4 live? | On the `ExtVTable`, so that they travel with the dtype and need no session lookup. Write them over `ReduceNode`, so that they also rewrite bound expressions for pruning |
| XQ3 | Do file reads check extension values, or trust the file? | Check only what memory safety needs, trust semantic constraints, and never panic on an illegal value. Offer a strict read mode that checks values |
| XQ4 | Which values can sit in null slots? | Any value. Accessors that take a child out of its parent mask it with the validity of the parent. This also allows a storage encoding to store nulls as a reserved value |
| XQ5 | How does an aggregate declare its result? | As a value of its input, as a bound on its input, or as neither. Only a value gets wrapped again |
| XQ6 | Does `Uuid` order by its bytes? | No recommendation yet |

## Decimal and new physical types

Issue [#10701](https://github.com/vortex-data/vortex/issues/10701) tracks the fix of the canonical
decimal encoding.

### Settled

| ID | Decision | Reason |
| --- | --- | --- |
| D1 | `Decimal` stays a built-in type. It behaves like a hardcoded extension type: opaque by default, with derives for one dtype and rules that the type owns. If the two become identical, a later change can remove the built-in type | A move to an extension type adds no capability, and the built-in type can follow the same rules |
| D2 | A canonical decimal array is a leaf with one buffer and validity. It has no data child | A leaf is simpler than a wrapper over an integer child, and it is the same as the `vortex.decimal` wire node |
| D3 | Each decimal dtype has exactly one canonical encoding. The precision fixes the width of the buffer | Today one dtype has six physical forms, which causes the crashes in issue [#5820](https://github.com/vortex-data/vortex/issues/5820) |
| D4 | A `Narrow` encoding stores a decimal column as a child with a lower precision and the same scale | Without it, D3 forces the full width on every column, which costs performance |
| D5 | Vortex adds `I128`, `I256`, `U128` and `U256` as primitive integer types | Wide decimals need an integer view, and wide integers are useful as columns |
| D6 | FixedSizeBinary becomes a core type for opaque bytes of a fixed length, such as UUIDs and hashes | Its defining operations (equality, hashing, byte order, cheap scalars) are core operations. Arrow and Parquet have it natively |
| D7 | `vortex.decimal` keeps its wire ID. A reader finds an old node when its stored width differs from the canonical width of its dtype | D3 removes forms and adds none. The versioning spec asks for a new ID only when readers must understand something new |
| D8 | One `Narrow` encoding, with one `vortex.narrow` wire ID, covers integers and decimals. Integers narrow by width (issue [#10020](https://github.com/vortex-data/vortex/issues/10020)), and decimals narrow by precision. The metadata stores only the child `PType` of an integer child, or only the child precision of a decimal child, in a prost message with a typed `oneof` (D32). It stores no scale and no nullability. This settles DQ2 and DQ3 | A release that permits an ID must read every valid instance of it. So the decimal reader ships in the same release as the first edition that permits `vortex.narrow`. The child must have the scale and nullability of the parent, so metadata that stored them would only let a file hold an invalid child. The format is in no edition yet, so it can still change |
| D9 | A reader converts an old `vortex.decimal` node. For a narrower node, it scans the non-null values for their max, and it chooses the best precision that the values and the scale allow, or a larger one. The result is a `Narrow` over a child at the width of that precision, or the canonical array when that width is the canonical width of the column. The result copies the buffer when its width differs from the stored width. For a wider node, the reader makes a narrowing copy (D21) | A width does not give a precision. The read plugin cannot use stored statistics, because the deserializer attaches them after the plugin returns. Old narrow nodes are rare, so the speed of this path matters little |
| D10 | A decimal `Narrow` child has the same scale and nullability as its parent, and a canonical width strictly narrower than the canonical width of the parent. The producer chooses the child precision: the best precision that the values and the scale allow, or a larger one | A `Narrow` that does not reduce the width has no use. The integer `Narrow` has the same rule. The rule can be relaxed later |
| D11 | The canonical-width work lands as an in-memory change, and then as a wire change (D18). Each step of the in-memory change is its own PR, because none of the steps needs temporary code. The wire change lands in lockstep, if possible in the same release as the in-memory change. PRs [#10305](https://github.com/vortex-data/vortex/pull/10305) and [#10314](https://github.com/vortex-data/vortex/pull/10314) stay open, and the work ignores them for now | Small PRs are easier to review, but not when a split needs temporary code or changes file contents twice. In one release, writers that select the preview edition never pay the `i256` cost of D17 |
| D12 | Each precision has the smallest signed width that holds its values: `i8` for precision 1 to 2, `i16` for 3 to 4, `i32` for 5 to 9, `i64` for 10 to 18, `i128` for 19 to 38 and `i256` for 39 to 76. `DecimalDType::canonical_values_type()` owns the mapping. Its name differs from the stored-width accessor `DecimalArray::values_type()`, so code cannot mix the two up. This settles DQ1 | It is the existing `smallest_decimal_value_type`. From precision 5, it is the width of the smallest Arrow decimal type that holds the precision |
| D13 | Imports copy into the canonical width. They write zeros in null slots, and they fail on a value outside the precision. The Arrow import logic can be configured to handle such values in another way. To widen an array, for example for an export, Vortex executes a `Narrow` with a wider dtype over it | arrow-rs does not check decimal values when it builds an array. An error is better than a silent change of the data |
| D14 | `DecimalArray::try_new` checks the width and every valid value, with an `ExecutionCtx` (D30). `new_unchecked` skips the value check, for kernels that keep values legal. File reads trust the precision and never panic | This is the rule of X7 for decimals, and the recommendation XQ3 for file reads |
| D15 | `Narrow` kernels pass work to the child, with canonicalization as the fallback. A `Narrow` over a `Narrow` becomes one `Narrow`. A cast of a `Narrow` passes to the child with the same target dtype, except that a wider integer target returns a `Narrow` over the child. Another decimal cast to a precision with the same scale returns a `Narrow` when the canonical width grows, changes only the dtype when the width stays, and checks every value when the precision drops | A `Narrow` holds exactly the values of its child, so work on the child gives the same result on less data |
| D16 | In DecimalByteParts, the MSP is any signed integer up to 64 bits, and every lower part has the dtype `u64`. The constructor and the module doc become stricter, and the constructor rejects lower parts that assemble wider than the canonical width (D29). This settles DQ4. Operations run on the parts wherever they can. Decoding produces the canonical array of the DecimalByteParts dtype. It widens a narrower MSP, and it narrows a wider MSP with the copy of D21. A `TODO(connor)` marks a later optimization: find the max of a narrow MSP, and return a `Narrow` that reuses its buffer | A narrow MSP needs no conversion at read time. The stricter constructor breaks no file, because no file has lower parts: no edition permits v2, and its compat fixture is disabled. The compressor never narrows the dtype of a part. Simple decoding is correct, and the optimization waits until benchmarks show large regressions |
| D17 | The compressor has one DecimalByteParts scheme. The scheme takes the canonical array, uses the value range to choose the number of lower parts, and narrows the MSP dtype as the compressor narrows every internal child. This replaces `narrowed_decimal` and keeps file contents, except that large values in null slots can give a narrower MSP. The scheme stays after the wire change, for writers whose edition does not permit `vortex.narrow`. When an edition permits `vortex.narrow`, a `Narrow` scheme runs before it and narrows the decimal type first (D33). In an edition without the lower-parts format, precision 39 to 76 with values that need more than 64 bits is written as a plain `i256` array. The `core` family holds the formats of the default writer, so default writers pay this cost until both formats enter `core` | The canonical fix makes a narrow decimal array illegal in memory, so the scheme cannot call `narrowed_decimal`. File contents do not change before the wire change. The `i256` cost affects only very wide decimals with values above 64 bits. The only way around it writes the old narrow form into new files, and it needs a special serializer |
| D18 | `vortex.narrow` stays in memory only until the maintainer decides to write it to files. Every new wire format enters a preview edition before `core`. `vortex.narrow` and DecimalByteParts with lower parts enter one preview edition together, and later `core` together. Other IDs can join that edition, but this plan adds only these two. This settles DQ9 | In-memory changes land first, so the edition change stays small |
| D19 | The bar of no compute regression applies to the final state, not to the steps in between. Missing benchmarks land before the changes that they cover. A gap column holds values that fit a width but need the precision of the next width. In the final state, such a column uses the wider width in memory, and that cost is accepted. Pushdown of sum, min and max into the MSP can come later | A true child label needs the next width for such values. Disk size and exports do not change |
| D20 | The canonical decimal work does not include the integer view, wide integers, FixedSizeBinary, decimal arithmetic across precisions and scales, casts that lose information, or the lower bound of a negative scale. Their recommendations are unreviewed future work. Epic [#10730](https://github.com/vortex-data/vortex/issues/10730) tracks wide integers and FixedSizeBinary. Issue [#9902](https://github.com/vortex-data/vortex/issues/9902) is its sub-issue for FixedSizeBinary, and wide integers get a tracking issue when that work is scheduled. The recommendation to always store min and max on the MSP (DQ8) is dropped | The canonical fix needs none of them. Stored min and max exist only on a BitPacked MSP, and a slice drops them |
| D21 | Every narrowing copy writes zeros in null slots, and it fails on a valid value outside the precision of the target dtype. Three paths make such a copy: an old node wider than its canonical width, an old narrow node whose best precision has a smaller width than the stored width, and DecimalByteParts decoding with an MSP wider than the canonical width. A read without a copy trusts the precision and does not scan | The copy reads every value anyway, so the precision check costs nothing extra, and it matches the import rule of D13. A check without a copy needs a scan that the read does not otherwise make |
| D22 | An integer `Narrow` child has the same signedness and nullability as its parent, and a strictly smaller width. So an `I64` column with values from 0 to 200 gets an `I16` child, not a `U8` child | Mixed signedness needs more code in casts, in the compare shortcut and in the arithmetic bounds, and it saves at most one width. Part 1 of PR [#10333](https://github.com/vortex-data/vortex/pull/10333) also keeps the signedness of internal buffers |
| D23 | `vortex.narrow` is only for columns with a fixed logical dtype: decimal columns and top-level integer columns. Which arrays the compressor wraps in a `Narrow` is deferred (D27). Internal integer children keep changing their own dtype, through the deprecated `encode_values` (D28). These include list offsets, dictionary codes, run ends, patch and sparse indices, `DateTimeParts`, VarBin and FSST offsets, and the DecimalByteParts MSP. Whether `List` and `ListView` offsets get a fixed `I64` dtype with `Narrow` is an open question on issue [#10733](https://github.com/vortex-data/vortex/issues/10733), and the decimal work does not depend on it | The parent chooses the dtype of an internal child, so the child can narrow its dtype without a `Narrow`. A column has a dtype that the user chose |
| D24 | The compat fixtures are published from a release before the PRs that change how files are read (D31), because only the old writer can make narrow leaves. They cover only reads that succeed: narrow, wide and Arrow-style widths, gap values, high scales at narrow widths, nonzero values in null slots, and DecimalByteParts with a narrow MSP. Error cases, such as a value outside the precision in a narrowing copy, are unit tests | The compat harness (`vortex-test/compat-gen`) expects the current reader to read every published fixture |
| D25 | Generic structural encodings always sit above `Narrow`. Reduce rules move `Dict`, `RunEnd`, `Sparse`, `Constant`, `Masked`, `Chunked` and `Interleave` above a `Narrow`. The chunked and interleave merge rules of the local rewrite, which put `Narrow` on top, are dropped. A follow-up issue covers the same rule for every encoding | The Arrow and DuckDB exporters and the dict layout look for these encodings at the root. One direction for every rule cannot loop |
| D26 | When a writer cannot write `vortex.narrow`, because no wire form exists yet or the selected editions do not permit it, the serializer writes the executed canonical array. The code documents this as a workaround. It stays until `vortex.narrow` is in `core` | Decimal `Narrow` appears in ordinary pipelines through casts and old files, and IPC or a writer without compression would fail on it. A plugin cannot return the wire form of another encoding, so the fallback lives in the serializer |
| D27 | Which arrays the compressor narrows is deferred to the design pass of WR3 | The candidates include struct fields, list elements, extension storage, and arrays that layouts create, such as dict-layout codes and zone maps |
| D28 | `Narrow` has `encode`, which builds a `Narrow`, and `encode_values`, which returns only the narrowed child with the same signedness. `encode_values` is `#[deprecated]`. It replaces `PrimitiveArrayExt::narrow` at every call site, and each call carries `#[expect(deprecated)]` with a `TODO(connor)` reason. The PR that moves the call sites checks that the latest release reads the files that it writes | Every narrowing that changes a dtype is then marked for a later decision. Readers on `develop` accept signed internal children |
| D29 | The DecimalByteParts constructor rejects a lower-part count whose assembled width is wider than the canonical width. Precision up to 18 allows no lower parts, 19 to 38 at most one, and 39 to 76 any count | Decoding then never needs a narrowing copy after assembly. No file breaks, because no edition permits lower parts |
| D30 | `DecimalArray::try_new` takes an `ExecutionCtx`, because the value check reads the validity, which can be lazy. Imports check their own values against the Arrow null buffer and then use the unchecked constructors, so a zero-copy import scans once | This follows `VarBinViewArray::try_new`, and it keeps the context out of the import API |
| D31 | Library changes land first. Only the PRs that change how files are read (DecimalByteParts decoding and the read plugin) wait for the compat fixtures, which the next regular release publishes. This changes D24 | The fixtures must be in the store when CI first runs the changed reader. Nothing else needs them |
| D32 | The `vortex.narrow` metadata is a prost message with a typed `oneof`: the child `PType` of an integer parent, or the child precision of a decimal parent. A missing variant is an error, and the `PType` decodes strictly | The typed `oneof` keeps the two kinds apart, and a missing variant cannot decode silently as `U8` |
| D33 | The compressor uses a real `Narrow` scheme at first. If the compressor turns out to always want it, it can become a hard-coded step. The design pass of WR3 settles its precedence with the DecimalByteParts scheme and the cascade level that it uses | A scheme fits the existing framework. The `AlwaysUse` contract and the cascade budget are the two costs to settle |
| D34 | Strict decoding covers the decimal width, the protobuf `PType` and the eleven `PType` fields of array and layout metadata, in one PR with three commits | It is the same bug with the same fix |
| D35 | The integer `Narrow` lands as five stacked PRs: the encoding, benchmarks, aggregates, arithmetic and casts, and the structural rules. The PR that moves the call sites to `encode_values` comes after them. The dictionary fill-code fix lands before the fifth PR | The benchmarks exist before the kernels change, and the fill-code fix guards the dictionary-codes rule |
| D36 | An operation on a `Narrow` fails only where the same operation on the parent dtype fails, and it gives the same result. Aggregates on a decimal `Narrow` use the accumulator of the parent dtype and feed it the child, so `Sum` and `SumV2` need no `Narrow` code | `Narrow` is a physical encoding, so it must not change behavior. A child accumulator overflows at a lower precision than the parent |
| D37 | The `vortex.decimal` read plugin converts old narrow nodes on the host, also on CUDA read paths. A GPU scan of such a node copies its buffer to the host | Old narrow nodes are rare, and CUDA has no `Narrow` kernel |

D1 to D6 date from 2026-10-19, D7 and D8 from 2026-10-22, and D12 to D37 from 2026-10-23. On
2026-10-23 the maintainer also revised D9, D10 and D11. Their 2026-10-22 versions used `P(W)`, the
largest precision whose canonical width is `W`, as the child precision of old data, and they
planned the work as one complete change. A second review on 2026-10-23 added D21 to D24, and it
changed D8, D9, D11, D16, D17 and D20. A planning review on the same day added D25 to D37, and it
changed D8, D12, D14, D15, D16, D17, D18, D23 and D24. [Part 2](decimal/02-design.md) of the Decimal
notes explains D7 to D37, and [part 4](decimal/04-implementation.md) splits the work into PRs.

### Open

These questions belong to future work (D20). Their recommendations are not reviewed.

| ID | Question | Recommendation |
| --- | --- | --- |
| DQ5 | Are wide integers user-visible columns, and which kernels do they get first? | User-visible behind an edition. A small kernel set first: compare, arithmetic, cast, sum, min, max, hash and structural operations |
| DQ6 | Does decimal arithmetic accept operands with different precision and scale? | Yes, with the Arrow rules for the result type. Comparisons keep the rule of equal dtypes |
| DQ7 | How does FixedSizeBinary appear on the wire? | As a new core dtype tag behind an edition |

## Casts

### Settled

| ID | Decision | Reason |
| --- | --- | --- |
| C1 | A cast between two dtypes exists only if a rule for that pair exists. Vortex ships rules for common pairs of built-in types, as the Rust standard library ships `From` implementations. Reading storage as another type is never a cast | Opt-in conversions with good defaults. Today casts reinterpret storage and disagree between paths |

### Open

| ID | Question | Recommendation |
| --- | --- | --- |
| CQ1 | What does the default cast do with a value that it cannot represent exactly? | It returns an error, like `TryFrom`. Lossy steps such as `trunc` and `round` are separate, named expressions. For decimals, this is future work (D20) |
| CQ2 | How do struct casts match fields? | By name, with unique field names (issue [#6900](https://github.com/vortex-data/vortex/issues/6900)). Extra source fields return an error |

## The `Narrow` pull requests

This review read the draft stack on 2026-10-19 and again on 2026-10-22. The state below is from
2026-10-23.

- The draft PRs [#10223](https://github.com/vortex-data/vortex/pull/10223),
  [#10332](https://github.com/vortex-data/vortex/pull/10332) and
  [#10333](https://github.com/vortex-data/vortex/pull/10333) date from 2026-10-06. The maintainer
  closed all three on 2026-10-23, and new PRs replace them. Issue
  [#10733](https://github.com/vortex-data/vortex/issues/10733) tracks the new PRs.
- A rewrite of #10223 and #10332 exists in five local commits, but it is not pushed. It makes
  `Narrow` an in-memory encoding with no serialized form. Since PR
  [#10681](https://github.com/vortex-data/vortex/pull/10681) merged on 2026-10-23, the rewrite
  needs a rebase. That PR removed `array_session()`, which the rewrite calls 47 times, and it
  conflicts with the rewrite in the test module of `dict/compute/fill_null.rs`. The local branch
  has the name `ct/narrow-pushdown`, which is the head branch of the closed #10332, so the new PRs
  use new branch names.
- The rewrite also fixes a fill code that the dictionary `fill_null` kernel truncates. That bug
  exists on `develop` without `Narrow`, so the fix becomes its own PR.
- The only integer-only check of #10223 and #10332 is the private `validate_dtypes`, so decimal
  support stays additive. When that check opens for decimals, the compare kernel and the sum
  pushdown need decimal code in the same change.
- The CodSpeed reports on #10223 and #10332 compare against `3c87da0`, because `cc018ee` had no
  run. Their gains, such as those of the FSST benchmarks, do not come from `Narrow`. The new PRs
  need new reports.
- The work of PR #10333 comes back in new PRs:
  1. NR1 adds `NarrowArray::encode_values`, and NR6 replaces `PrimitiveArrayExt::narrow` with it at
     every call site (D28). Internal buffers then keep their signedness in every preset. Golden
     snapshots and file contents change, but no wire format changes.
  2. The serialization of D8, the edition entry and the `Narrow` scheme in the compressor are part
     of the wire change, WR1 to WR3 (D11, D18).
- [Part 4](decimal/04-implementation.md) of the Decimal notes lists every new PR, and the briefs in
  `decimal/agents/briefs/` describe each one.
- The CodSpeed report on #10333 shows `take_map` regressions of 22% and 11% in Simulation mode.
  The `take_map` benchmark builds `u64` patch indices directly and does not call the narrowing
  code. The cause is not known. The new PRs get fresh CodSpeed runs, and the investigation
  happens there.
- PR [#10230](https://github.com/vortex-data/vortex/pull/10230) gave decimals an integer child,
  which D2 replaces. It closed on 2026-10-19. Its fix to the DecimalByteParts cast rule becomes its
  own PR, with no issue. The PR adds the guard of #10230, a cast kernel for the case that needs
  execution, and tests. [Part 3](decimal/03-roadmap.md) of the Decimal notes replaces the other
  steps of #10230.
- PR #10305 limits the decimal storage width to the canonical width, and PR #10314 adds bounded
  dispatch macros on top of it. Another contributor owns both, and both are open. The work
  ignores them for now (D11).
