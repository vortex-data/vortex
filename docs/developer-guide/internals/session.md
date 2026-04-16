# Session and Registries

A `VortexSession` is a type-indexed map that holds all the runtime state for the Vortex
ecosystem. Each major component of Vortex defines its own session variable type containing a
registry of plugins. There is typically one session per process, and it is passed explicitly
through the API rather than relying on global or thread-local state.

Plugins are themselves also able to register state in the session, for example to hold caches
or other shared resources.

## Design

The session is built on two primitives from the `vortex-session` crate:

- **`VortexSession`** -- a cloneable, thread-safe map from Rust `TypeId` to a boxed value. Any
  type that is `Send + Sync + Debug + 'static` can be stored as a session variable.
- **`Registry<T>`** -- a concurrent map from string IDs to values of type `T`, used by each
  component to look up registered plugins at runtime.

Because `VortexSession` is backed by an `Arc<DashMap>`, cloning is cheap and all clones share
the same state. This makes it safe to hand the session to multiple threads, tasks, or I/O
operations without coordination.

## Component Registries

Each Vortex crate defines a session variable that holds a registry for its extension points:

| Session Variable  | Crate            | Registry Contents                            |
|-------------------|------------------|----------------------------------------------|
| `DTypeSession`    | `vortex-array`   | Extension dtype vtables (Date, Time, ...)    |
| `ArraySession`    | `vortex-array`   | Array encoding vtables (ALP, FSST, ...)      |
| `ScalarFnSession` | `vortex-array`   | Scalar function vtables                      |
| `LayoutSession`   | `vortex-layout`  | Layout encoding vtables (Flat, Chunked, ...) |
| `RuntimeSession`  | `vortex-io`      | Async runtime handle                         |
| `CudaSession`     | `vortex-cuda`    | CUDA context, kernels, and stream pool       |

Session variables are created lazily on first access with their `Default` implementation, which
registers the built-in plugins for that component. For example, `ArraySession::default()`
registers the 14 built-in encodings (Null, Bool, Primitive, Struct, etc.), and
`LayoutSession::default()` registers the 5 built-in layouts (Flat, Struct, Chunked, Zoned, Dict).

## Registering Plugins

Plugins register with the session by accessing the relevant component and calling `register`:

```rust
// Register a custom array encoding
session.arrays().register(MyEncoding);

// Register a custom layout
session.layouts().register(MyLayout::encoding());

// Register a custom scalar function
session.scalar_fns().register(MyScalarFnVTable);
```

Crates that bundle multiple plugins typically expose an `initialize` function that registers
everything at once. The top-level `vortex` crate calls these during `VortexSession::default()`
to register all built-in encodings.

## Explicit Passing

Sessions are passed explicitly through constructors and method arguments. This means every API
that needs access to registries -- file readers, writers, scan builders, layout readers -- receives
the session directly rather than reaching for global state.

```rust
// Opening a file
session.open_options()
    .open(reader)
    .await?;

// Writing a file
session.write_options()
    .write(&mut file, array_stream)
    .await?;

// Scanning a layout
ScanBuilder::new(session.clone(), layout_reader)
    .with_filter(expr)
    .into_array_stream()?;
```

Many APIs use extension traits to provide ergonomic methods directly on the session. For example,
`OpenOptionsSessionExt` adds `.open_options()` to any session that has `ArraySession`,
`LayoutSession`, and `RuntimeSession` registered. This lets the type system enforce that the
required components are present.

## Constructing a Session

The `VortexSession::default()` provided by the top-level `vortex` crate constructs a session with
all built-in components and encodings:

```rust
let session = VortexSession::default();
```

For tests or specialized use-cases, sessions can be assembled from individual components using
the `.with::<T>()` builder:

```rust
let session = VortexSession::empty()
    .with::<ArraySession>()
    .with::<LayoutSession>()
    .with::<ScalarFnSession>()
    .with::<RuntimeSession>();
```
