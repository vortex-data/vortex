# Dispatch Macro Guidance

Applies to files under `vortex-array/src/dtype/`, and to every call of the dispatch macros declared
here.

## Dispatch macros

The `match_each_*!` macros turn a runtime type tag into a type parameter. The ptype families
(`match_each_native_ptype!`, `match_each_integer_ptype!`, `match_each_unsigned_integer_ptype!` and
the others) are declared in `ptype.rs`, and the decimal ones in `decimal/macros.rs`. The same rules
apply to the other dispatch macros in `vortex-array`: `match_each_pvalue!`
(`src/scalar/typed_view/primitive/pvalue.rs`), `match_each_canonical!` (`src/canonical.rs`), and
the builder macros in `src/builders/mod.rs` and `src/arrays/varbin/builder.rs`.

cargo-mutants cannot mutate code inside a macro invocation or a `macro_rules!` body, so logic
written there is never mutation tested. Keep each call a thin dispatch: resolve the type, then
call a generic function that holds the logic.

```rust
// Logic inside the macro body is invisible to cargo-mutants.
match_each_integer_ptype!(array.ptype(), |T| {
    let mut total = 0u64;
    for &value in array.as_slice::<T>() {
        total += value.as_();
    }
    total
})

// The macro only dispatches; `sum_values` can be mutated.
match_each_integer_ptype!(array.ptype(), |T| sum_values(array.as_slice::<T>()))

/// Sums `values` as `u64`.
fn sum_values<T: AsPrimitive<u64>>(values: &[T]) -> u64 {
    let mut total = 0u64;
    for &value in values {
        total += value.as_();
    }
    total
}
```

- Use a plain generic function with static dispatch. Do not add `dyn`, boxing, allocations or
  copies: it must monomorphize to the same code as the inline body.
- Pass the slices and values that the body used, and keep the function private unless other
  modules need it.
- A call that only forwards to one function, or a `match` whose arms only forward, is already thin
  dispatch.
- When writing a new `macro_rules!` macro, keep it to dispatch or boilerplate as well, and put the
  logic in the functions it calls.
