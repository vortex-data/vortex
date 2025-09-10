# RFC: Hierarchical Error Handling with Snafu

Status: Proposed

## Summary

This RFC proposes migrating Vortex's error handling from a centralized `VortexError` enum to a hierarchical error system using the [`snafu`] library. The migration would proceed incrementally, with each crate defining its own error types that convert into the existing `VortexError`, allowing for a gradual transition without breaking changes. Once the migration is done, we can remove the current `VortexError` and replace it with a top-level, user-facing `VortexError`.

[`snafu`]: https://docs.rs/snafu/latest/snafu/

## Motivation

Vortex currently uses a single `VortexError` enum shared across the entire codebase. While this approach is relatively consistent and easy, it has several limitations that has clearly become increasingly problematic.

### Current Pain Points

When debugging failures, we will likely come across generic error variants like `WithContext` that really only provide limited context about what has actually happened. The error types don't encode which operations or modules can produce which errors, making it difficult to handle specific failure cases programmatically. Of course, it is very possible for us to manually inspect the backtrace, but in my opinion this is not ideal, and we can definitely do better.

Some other (small) things: The 128-byte size limit on `VortexError` constrains how much context can be stored, and the manual backtrace capture adds runtime overhead even when not needed (and I would argue that most of the time we don't _really_ need it, see the [#GreptimeDB] section and the [blog post](https://greptime.com/blogs/2024-05-07-error-rust)).

Basically, the current system makes it difficult to understand error propagation paths through the _entire_ codebase, as any function returning `VortexResult` could theoretically produce any variant of `VortexError`.

I found that the the current `VortexError` also causes a lot of confusion on the developer side as to what error variant to return. I didn't even know that there were so many variants until recently, since the entirety of the code that I have interacted with over the past 3 weeks simply uses `vortex_bail` or `vortex_err!`, which returns a `WithContext` variant with an error string. This is the **exact** definition of a strawman approach to error handling, and it completely disregards the benefits of strong type consistency that Rust provides us.

### Benefits of Hierarchical Errors

A hierarchical error system would establish clear ownership boundaries, with each module maintaining its own error types that accurately reflect its failure modes. This approach provides several key advantages, and is basically taken from GreptimeDB and their [blog post](https://greptime.com/blogs/2024-05-07-error-rust).

Error context would be automatically captured at each level of the call stack, creating a "virtual stack trace" that shows the logical flow of operations rather than just the final failure point. Module-specific error types would make it immediately clear which operations can fail in which ways, improving both API documentation and code comprehension. As a small benefit, the workspace dependency tree would be inverted with respect to errors, with the lower-level crates no longer depending on the `vortex-error` crate.

The snafu library provides ergonomic macros for error creation and context addition, reducing boilerplate while maintaining type safety. I would say it is a mix between `thiserror` and `anyhow`, but my personal experience using `snafu` has been quite different than just smashing those two libraries together.

## Detailed Design

### Error Hierarchy Structure

Each crate would define its own error module with types specific to its domain. For example, `vortex-buffer` would define errors related to memory alignment and bounds checking, while `vortex-array` would have errors for type mismatches and compute operations. These error types would form a natural hierarchy following the module structure of each crate. This is the philosophy that `snafu` tries to cater to.

```rust
// vortex-buffer/src/error.rs
use snafu::{Snafu, Location, Backtrace};

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum BufferError {
    #[snafu(display("Buffer alignment error: required {required}, found {actual}"))]
    Alignment {
        required: usize,
        actual: usize,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Slice range {start}..{end} out of bounds for buffer of length {len}"))]
    SliceOutOfBounds {
        start: usize,
        end: usize,
        len: usize,
        location: Location,
        backtrace: Backtrace,
    },
}
```

### Context Propagation

Snafu's context selectors would replace the current `vortex_err!` and `vortex_bail!` macros. When an error crosses module boundaries, context would be added to track the operation that failed:

```rust
// vortex-array/src/compute/cast.rs
use snafu::{ResultExt, Snafu, Location, Backtrace};

#[derive(Debug, Snafu)]
enum CastError {
    #[snafu(display("Cannot cast from {from} to {to}"))]
    IncompatibleTypes {
        from: DType,
        to: DType,
        location: Location,
        backtrace: Backtrace,
    },

    #[snafu(display("Buffer operation failed during cast"))]
    BufferOperation {
        #[snafu(source)]
        source: vortex_buffer::BufferError,
        location: Location,
        backtrace: Backtrace,
    },
}

fn cast_array(array: &Array, target: DType) -> Result<Array, CastError> {
    let buffer = array.buffer()
        .context(BufferOperationSnafu)?;
    // Cast implementation
}
```

### Backward Compatibility

During the migration period, each crate's error type would implement `From` for conversion to `VortexError`. This allows existing code to continue working unchanged while new code can leverage the richer error types:

```rust
impl From<vortex_buffer::BufferError> for VortexError {
    fn from(error: vortex_buffer::BufferError) -> Self {
        // Map specific buffer errors to appropriate VortexError variants
        match error {
            vortex_buffer::BufferError::SliceOutOfBounds { start, end, len, .. } => {
                VortexError::OutOfBounds(
                    format!("Buffer slice {start}..{end} exceeds length {len}"),
                    Backtrace::capture()
                )
            }
            _ => VortexError::Generic(error.to_string(), Backtrace::capture())
        }
    }
}
```

Or even better, since `VortexError` is `#[non_exhaustive]`, we can just throw more variants onto there that wrap these new error types. It's not like we're really making an effort to keep the number of variants down in the first place...

### Error Documentation

Each error variant would include documentation explaining when it occurs and how to handle it. The `snafu` derive macro automatically generates helpful error messages that include all relevant context:

```rust
#[derive(Debug, Snafu)]
pub enum CompressionError {
    /// Returned when the input data exceeds the maximum size supported
    /// by the compression algorithm (typically 2GB for 32-bit encodings).
    #[snafu(display("Input size {size} exceeds maximum {max}"))]
    InputTooLarge {
        size: usize,
        max: usize,
        location: Location,
        backtrace: Backtrace,
    },
}
```

In my opinion, this is SO much better than manually creating this format string EVERY single time we call `vortex_err!` or `vortex_bail!`.

_From here on out, everything is mostly Claude-generated, though I have heavily edited it. I give my commentary here and there. If this first section above passes the vibe check, I'll go through the next part in more detail and make sure things make sense._

## Drawbacks and Risks

_Honestly, I don't think any of these drawbacks are worth sitting on when the error system right now is not great._

### Binary Size Increase

While GreptimeDB reported only 100KB increase for their entire system, Vortex's extensive use of generics and monomorphization could lead to larger binary size increases. Each instantiation of generic functions with different error types could generate additional code. The embedded location information and error strings would add to binary size even if never triggered.

### Compilation Time Impact

The snafu derive macros add to compilation time, though this is generally negligible compared to other procedural macros already in use. More concerning is that changing an error type in a low-level crate would trigger recompilation of all dependent crates, though this is somewhat mitigated by the incremental compilation cache.

### Migration Risks

During the transition period, the codebase would have two different error handling patterns in use simultaneously. This could lead to confusion and mistakes. There's risk of introducing subtle bugs during the conversion, particularly around error propagation boundaries. Test coverage would be critical to catch these issues.

### API Stability

The migration would eventually require breaking changes to public APIs as error types change. While the `From` implementations provide backward compatibility initially, eventually users would want to handle specific error types, requiring API updates.

## Prior Art

### `snafu`

See the docs for [`snafu`], as well as the accompanying [philosophy](https://docs.rs/snafu/latest/snafu/guide/philosophy/index.html).

### GreptimeDB

GreptimeDB successfully implemented a hierarchical error handling system using snafu, as detailed in their blog post ["Error Handling for Large Rust Projects"](https://greptime.com/blogs/2024-05-07-error-rust). They reported several key benefits from their migration:

- Only 100KB binary size increase despite adding comprehensive error context throughout their codebase
- Virtual stack traces that provide logical operation flow rather than system-level backtraces
- Improved debugging efficiency with contextual error information at each layer
- Better error messages for end users through structured error propagation
- Easy migration path using the `#[stack_trace_debug]` macro on existing error types

Their approach of creating per-module error types that form a tree structure directly inspired this RFC. They also demonstrated that migration can be incremental—existing error types can be annotated with `#[stack_trace_debug]` to add location tracking without requiring a full rewrite, allowing teams to adopt the pattern gradually. GreptimeDB's experience demonstrates that this pattern scales well for large Rust projects with complex error propagation requirements.

### `iroh`

They wrote a blog about using `snafu` as well: [blog](https://www.iroh.computer/blog/error-handling-in-iroh). They argue that an even more fine-grained approach is necessary for complex code, and instead of using module-level errors they have _function_-level errors.

### Other Projects Using Snafu

- **[Vector](https://github.com/vectordotdev/vector)**: A high-performance observability data pipeline that uses snafu throughout their codebase to maintain error context across transformations. Their [error handling patterns](https://github.com/vectordotdev/vector/search?q=snafu) show extensive use of contextual errors across dozens of modules.

- **[pdf-rs](https://github.com/pdf-rs/pdf)**: A Rust library for reading, manipulating, and writing PDF files that uses snafu for error handling.

While snafu may not be as widely adopted as thiserror or anyhow, its use in these production systems—particularly in data-intensive applications like GreptimeDB and Vector—demonstrates its effectiveness for managing complex error hierarchies in large Rust codebases where context preservation is critical.

## Alternatives Considered

### Enhanced Central Error Type

We could keep the centralized approach but enhance `VortexError` with better context tracking. This would avoid the migration complexity but wouldn't address the fundamental coupling issues or provide module-specific error types.

### anyhow/color-eyre

These libraries provide excellent error context, but are better suited for applications than libraries. This would really just be replacing the current system with the same issues.

### `thiserror` Without Hierarchy

We could use thiserror to reduce boilerplate while keeping a flat error structure. This would provide some ergonomic improvements but wouldn't address the core architectural issues around error ownership and context.

## Testing Strategy

Each migrated crate would need comprehensive error path testing. Property-based testing could ensure error messages contain expected context. Integration tests would verify error propagation across crate boundaries. Benchmarks would measure any performance impact on error-heavy workloads.

The existing test suite's 389 test modules would need updates to handle new error types. This could be done incrementally as each crate is migrated.

## Conclusion

Migrating to hierarchical error handling with snafu represents a significant architectural improvement for Vortex. While the migration effort is substantial, the incremental approach allows us to validate benefits early and abort if needed. The improved debugging experience, clearer module boundaries, and better error context would provide long-term value that justifies the investment.

The key to success lies in careful planning, incremental execution, and maintaining backward compatibility throughout the transition. By starting with leaf crates and working up the dependency tree, we can minimize risk while continuously delivering value.

---

## Implementation Plan

### Phase 1: Infrastructure Setup

The first phase would establish the foundation for the migration without changing any existing error handling. We would add snafu as a workspace dependency and create an error conversion trait that all new error types would implement. A set of guidelines and examples would be documented for consistent error design across crates. Helper macros might be created to reduce boilerplate in the `From` implementations.

This phase primarily involves planning and setup work. The main challenge lies in designing patterns that will scale across all 44 crates while maintaining consistency.

### Phase 2: Leaf Crate Migration

The migration would begin with crates that have no dependencies on other workspace crates. These leaf crates, such as `vortex-buffer`, `vortex-dtype`, and `vortex-scalar`, provide the simplest starting point. Each would receive its own error module with types specific to its domain.

These crates have relatively simple error scenarios, making them ideal for establishing patterns and training the team. The conversion is straightforward since they don't need to handle errors from other workspace crates.

### Phase 3: Encoding Libraries

The 13 encoding libraries (`alp`, `bytebool`, `dict`, `fsst`, etc.) would be migrated next. These crates are relatively independent of each other but depend on the core types migrated in Phase 2. Each encoding would define errors specific to its compression and decompression operations.

The encoding libraries have moderate complexity with well-defined error boundaries. The main challenge involves handling errors from the underlying buffer and type system crates while maintaining clean abstractions.

### Phase 4: Core Array System

The `vortex-array` crate represents the most complex migration challenge. With 275 source files organized into 46 subdirectories, it would require careful planning. The error hierarchy would mirror the module structure, with separate error types for compute operations, array implementations, and pipeline stages.

This phase is significantly more complex than earlier ones. The interconnected nature of array operations means careful attention must be paid to error propagation paths. The benefit is that once complete, this provides a solid foundation for the remaining crates.

### Phase 5: I/O and File Systems

The I/O layer (`vortex-io`, `vortex-file`) would be migrated to properly handle and propagate errors from external sources like object stores and file systems. These errors would maintain the context of what operation was being attempted when the I/O failure occurred.

This phase has moderate complexity but requires careful handling of external error types. The async nature of much of the I/O code adds an additional consideration for error handling.

### Phase 6: Integration Layers

Language bindings (Python, Java, C++) and the DataFusion integration would be updated to handle the new error types. Each binding would need strategies for converting rich Rust errors into appropriate representations for the target language.

The FFI boundaries make this phase particularly challenging. Each language binding has different constraints on how errors can be represented, requiring custom conversion logic.

### Phase 7: Cleanup and Optimization

Once all crates have been migrated, the original `VortexError` enum would be refactored into a thin facade that primarily serves as a top-level error type for public APIs. The old error macros would be removed, and documentation would be updated to reflect the new patterns.

This phase is primarily mechanical but requires careful coordination to ensure nothing breaks during the transition.
