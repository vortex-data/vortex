# Writing a Vortex encoding

Work in progress.

## Statistics and memory safety

Statistics are hints. They can come from untrusted input: a serialized array stores statistics
for each node, and decoding installs them on the array without a check against the data.

- **Logical results can depend on statistics.** A compute function can use a statistic to
  short-circuit or to select a faster path. A forged statistic then gives a wrong result, but not
  undefined behavior.
- **Memory safety must not depend on statistics.** Do not read the statistics of an array or its
  children in `VTable::validate`, or to make a decision that an `unsafe` block relies on. Check
  the invariant on the data.

## When to validate

Validate an invariant at the latest point where the encoding can still give a meaningful result
for data that does not satisfy it.

- **Validate at construction when records depend on each other.** `List` and `VarBin` find each
  record from two adjacent offsets, so one bad offset changes the adjacent records. These
  encodings cannot process mismatched data, and they must check the offset values before the
  array is published.
- **Validation can be deferred when records are independent.** `ListView` and `VarBinView` give
  each record its own offset and size, so a bad record does not change other records. Such an
  encoding can check a record when it processes the record, and report an error for that record
  only. This keeps construction cheap. `Dict` defers validation: each code refers to one
  dictionary value independently, so `Dict` does not check the range of the codes at
  construction. The check occurs when the codes are used to take from the values.
- **Validate early when later operations need the invariant.** A check for each record prevents
  SIMD and other branch-free loops. If the array is validated one time at construction, later
  operations can process the records without checks. `VarBinView` validates its views at
  construction for this reason, although its records are independent.

Deferred validation is still validation of the data. The check must run before the access that
it protects, and it must not read statistics.
