<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Decisions

This file records the decisions that the maintainer made while reviewing these notes on
2026-10-19. It also lists the questions that stay open. The other documents follow this file. If a
document disagrees with it, this file wins.

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

### Settled

| ID | Decision | Reason |
| --- | --- | --- |
| D1 | `Decimal` stays a built-in type. It behaves like a hardcoded extension type: opaque by default, with derives for one dtype and rules that the type owns. If the two become identical, a later change can remove the built-in type | A move to an extension type adds no capability, and the built-in type can follow the same rules |
| D2 | A canonical decimal array is a leaf with one buffer and validity. It has no data child | A leaf is simpler than a wrapper over an integer child, and it is the same as the `vortex.decimal` wire node |
| D3 | Each decimal dtype has exactly one canonical encoding. The precision fixes the width of the buffer | Today one dtype has six physical forms, which causes the crashes in issue [#5820](https://github.com/vortex-data/vortex/issues/5820) |
| D4 | A `Narrow` encoding stores a decimal column as a child with a lower precision and the same scale | Without it, D3 forces the full width on every column, which costs performance |
| D5 | Vortex adds `I128`, `I256`, `U128` and `U256` as primitive integer types | Wide decimals need an integer view, and wide integers are useful as columns |
| D6 | FixedSizeBinary becomes a core type for opaque bytes of a fixed length, such as UUIDs and hashes | Its defining operations (equality, hashing, byte order, cheap scalars) are core operations. Arrow and Parquet have it natively |

### Open

| ID | Question | Recommendation |
| --- | --- | --- |
| DQ1 | Which width does each precision get? | The smallest signed width that holds the digits, which is the existing mapping in `vortex-array/src/dtype/decimal/types.rs:46-56` |
| DQ2 | Is `Narrow` one encoding for integers and decimals? | Yes. Integers narrow by width (issue [#10020](https://github.com/vortex-data/vortex/issues/10020)). Decimals narrow by precision |
| DQ3 | Does decimal `Narrow` share the wire ID `vortex.narrow`? | No, unless decimal `Narrow` lands before the edition of PR #10333 ships in a release. A second wire ID needs no change to the open PRs |
| DQ4 | Does the dtype of a DecimalByteParts array fix the type of its most significant part? | Yes. Narrowing then happens only through `Narrow` and integer encodings. A reader wraps the narrow part of an old file in an integer `Narrow` of the fixed type, without a copy |
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
| CQ1 | What does the default cast do with a value that it cannot represent exactly? | It returns an error, like `TryFrom`. Lossy steps such as `trunc` and `round` are separate, named expressions |
| CQ2 | How do struct casts match fields? | By name, with unique field names (issue [#6900](https://github.com/vortex-data/vortex/issues/6900)). Extra source fields return an error |

## The open `Narrow` pull requests

This review read the draft stack on 2026-10-19.

- PR [#10223](https://github.com/vortex-data/vortex/pull/10223) and PR
  [#10332](https://github.com/vortex-data/vortex/pull/10332) can merge as they are. Their only
  integer-only check is the private `validate_dtypes`, so decimal support stays additive. When
  that check opens for decimals, the compare kernel and the sum pushdown need decimal code in the
  same change.
- PR [#10333](https://github.com/vortex-data/vortex/pull/10333) adds `vortex.narrow` to the draft
  edition `preview2026.10.0`, with one metadata byte that holds the child `PType`. Question DQ3
  decides it before the merge. The editions spec says that a change to what readers must
  understand needs a new wire ID (`docs/specs/versioning/editions.md:118-121`).
- PR [#10230](https://github.com/vortex-data/vortex/pull/10230) gives decimals an integer child,
  which D2 replaces. Split its reusable parts into smaller PRs, in this order:
  1. The nullability fix in DecimalByteParts.
  2. Producers take the width from the precision.
  3. `Narrow` over decimals.
  4. The compressor and DecimalByteParts decoding stop narrowing and return `Narrow` instead.
  5. Readers convert old buffers at other widths.
  6. `DecimalArray::try_new` accepts only the canonical width.

  Decimal kernels can move onto integer kernels through a zero-copy view at any point.
