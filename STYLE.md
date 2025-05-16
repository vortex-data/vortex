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
- Run `cargo fmt` before submitting code

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
- Use `VortexExpect` and `VortexUnwrap` traits when unwrapping is appropriate

## Code Structure

- Maintain a clear separation between logical and physical types
- Keep functions focused and reasonably sized
- Separate public API from internal implementation details
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

## Linting

- Run `cargo clippy --all-targets --all-features` before submitting code
- Resolve all clippy warnings
- Follow custom clippy configuration:
  - Single character binding names threshold of 2
  - Avoid disallowed types like `HashMap` and `HashSet`
