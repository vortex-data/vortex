# Vortex Code Style Guide

## General Principles

- Write clean, readable, and maintainable code
- Follow standard Rust idioms and best practices
- Prioritize safety and correctness
- Prefer zero-cost abstractions where possible
- Minimize use of `unsafe` to cases where it's truly necessary (i.e., typically when the performance benefits are large)

## Code Formatting

- Use `rustfmt` with the project's custom configuration:
  - Condense wildcard suffixes
  - Format macro matchers and bodies
  - Group imports by StdExternalCrate
  - Use field init shorthand
  - Group imports at the module level
  - Use 2024 edition style
- Run `cargo +nightly fmt` before submitting code

## Documentation

- Every public API definition MUST have a doc comment
- Module-level documentation using `//!` comments for context and purpose
- Function-level documentation using `///` comments
- Examples in documentation are encouraged but not strictly required
- Use `#![deny(missing_docs)]` in crates to enforce documentation standards

## Naming Conventions

- Follow standard Rust naming conventions:
  - `CamelCase` for types, traits, and enums
  - `snake_case` for functions, methods, and variables
  - `SCREAMING_SNAKE_CASE` for constants and statics
- Use descriptive names that clearly convey purpose
- Prefer explicit names over overly terse abbreviations

## Type System

- Prefer strongly typed APIs when possible
- Use Rust's type system to prevent bugs at compile time
- Implement appropriate traits for custom types
- Prefer `impl AsRef<T>` to `&T` for public APIs (e.g. `impl AsRef<Path>`)
- Use type aliases to improve code readability and maintenance

## Error Handling

- Use the custom `VortexError` type for errors
- Propagate errors using the `?` operator
- Use the following error macros consistently:
  - `vortex_err!` for creating errors
  - `vortex_bail!` for returning errors
  - `vortex_panic!` for handling invariant violations
- Add context to errors using `.with_context()`
- Include backtraces for better debugging
- Use `VortexExpect` trait when unwrapping is appropriate with proper error context.

## Code Structure

- Maintain a clear separation between logical and physical types
- Keep functions focused and reasonably sized
- Separate public API from internal implementation details
- Prefer one public entrypoint for each piece of functionality; keep helper APIs crate-private
  unless callers need them independently.
- Use modules to organize related functionality
- Place tests in a `tests` module or separate test files

## Collections and Data Structures

- Avoid using `HashMap` and `HashSet` from the standard library (prefer the alternatives in `vortex-array::aliases`)
- Prefer specialized collections when appropriate
- Be mindful of performance implications when choosing data structures

## Safety and Unsafe Code

- Avoid `unsafe` code unless strictly necessary for optimal performance
- Document all uses of `unsafe` with detailed safety comments
- Encapsulate `unsafe` code within safe abstractions

## Testing

- Write comprehensive unit tests for new functionality
- Include integration tests for complex features
- Use property-based testing for appropriate scenarios
- Follow test naming conventions: `test_<function_name>_<scenario>`
- In tests only:
  - `dbg!` usage is allowed
  - `expect()` and `unwrap()` are acceptable
  - More relaxed clippy rules apply

## Dependencies

- Be conservative with adding new dependencies
- Follow dependency management guidelines in `deny.toml`
- Prefer using crates from the workspace when possible

## Performance Considerations

- Optimize for readability & performance (choose two)
- Use benchmarks to measure performance improvements
- Prefer algorithmic improvements over micro-optimizations
- Document performance-critical sections

### Avoid Hidden-Cost Accessors in Hot Loops

Do not call a per-element accessor that hides non-trivial work inside an `O(n)` loop. Each call can
repay work that is constant or amortizable across the chunk, turning the loop into `O(n * k)`.

Watch for these accessors inside `for i in 0..n { ... }`:

| Per-element accessor | Hidden cost | Bulk replacement |
| --- | --- | --- |
| `Validity::is_valid(i)` / `is_null(i)` | Array-backed validity allocates an `ExecutionCtx` and runs a scalar lookup per call. | Call `validity.execute_mask(len, ctx)?` once, then read the materialized mask. |
| `array.scalar_at(i)` / `array.execute_scalar(i, ctx)` | Executes through the compute stack per element. | Canonicalize once with `execute::<PrimitiveArray>` or `as_slice`, then index. |
| `BitBuffer::value(i)` / `Mask::value(i)` accumulated into a count | Recomputes the byte address and defeats popcount. | Use `true_count()`, `BitBuffer::count_range(start, end)`, or `set_indices()`. |
| `BitIterator::next()` accumulated into a rank or prefix count | Processes one bit at a time. | Use `count_range` over each gap. |
| Re-deriving a value such as `self.validity()?` | Repeats the derivation for every element. | Hoist the derivation above the loop. |

Choose the replacement based on the access pattern:

- For sequential or contiguous access, materialize once and iterate or index the chunk.
- For a gather over arbitrary indices, materialize the backing buffer once, then use cheap random
  reads. The decode itself may not be amortizable.
- Leave genuinely `O(1)` accessors alone. Bulk materialization does not help an already materialized
  mask, slice, or native bitmap.

After materializing a `Mask` or `BitBuffer`, avoid calling `value(i)` for every element merely to act
on set bits. Use `BitBuffer::for_each_set_index`, which iterates words with all-set and all-unset fast
paths. Use cached `indices()` or `slices()` representations when they will be reused.

Back changes to these loops with an appropriate benchmark:

- `vortex-array/benches/validity_is_valid.rs` for validity access.
- `vortex-mask/benches/valid_counts.rs` for popcount.
- `vortex-mask/benches/mask_iteration.rs` for set-bit iteration.

## Linting

- Run `cargo clippy --all-targets --all-features` before submitting code
- Resolve all clippy warnings
- Follow custom clippy configuration:
  - Single character binding names threshold of 2
  - Avoid disallowed types like `HashMap` and `HashSet`
