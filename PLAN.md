# Expressions Branch Plan

## Intent

`ngates/expressions` is the long-lived integration branch for the expressions work. The
expected scale is large, roughly 50k lines, and API churn is expected while the design is
being discovered.

The goal is not to preserve a perfectly linear development history during exploration.
The goal is to make forward progress on the integration branch while preserving enough
structure, checkpoints, and notes that the final result can be split into reviewable PRs.

This file is the operating agreement for that work. Update it when the process changes,
when major decisions settle, or when likely PR boundaries become clearer.

## Working Model

- Treat `ngates/expressions` as the durable integration branch.
- Work may arrive through direct commits, topic branches, merges, cherry-picks, or agent
  changes. The workflow does not need to be linear.
- Do not force-push, reset, or rewrite the long-lived branch unless explicitly requested.
- Prefer merging current `develop` into `ngates/expressions` at stable checkpoints rather
  than rebasing the shared branch.
- Keep the branch pushable and recoverable. It may be temporarily broken during active
  API migration, but stable checkpoints should build and should be easy to identify.
- Use separate throwaway/topic branches when that helps isolate experiments, but merge
  the useful result back into `ngates/expressions`.
- Big-bang migrations and temporary API breakage are acceptable on this branch when they
  produce a clearer final design. Do not add elaborate long-term compatibility shims
  just to avoid churn during the migration.

## Current Design Direction

Vortex should grow a high-level expression DSL that behaves like the function catalog in
other query engines. Users should be able to write friendly function names such as
`contains`, while Vortex resolves the call to the correct implementation, such as
`geo.contains` or `list.contains`, based on argument types, overloads, and coercion rules.

The DSL is not intended to become a general query planner. It presents Vortex as an
engine whose only logical plan node is a scan. The high-level surface therefore needs
expressions, scalar functions, aggregate functions, binding, coercion, and scan
construction, but not a full tree of relational operators or plan nodes.

The existing low-level expression tree is the already-bound representation. It currently
stores every node as a scalar function plus options and children. The intended direction
is to rename or reframe this API as the lower-level bound API, likely `BoundExpr`, but
the internal shape is open to change. Callers such as `ScanBuilder` should accept
`BoundExpr` once the function overloads and coercions have been resolved.

High-level language APIs such as Python and Java should expose the friendly DSL by
default. They may also expose the lower-level bound API for advanced users, engine
integrations, and tests that need exact function identity.

## Naming Candidates

These names are provisional. Prefer names that make the layering obvious:

- High-level unbound expression: `Expr`, `DslExpr`, or `UnboundExpr`.
- Low-level bound expression: `BoundExpr` is the current preferred name for today's
  `Expression`.
- Catalog/session object: `FunctionCatalog`, `DslCatalog`, or `VortexCatalog`.
- Friendly overload group: `FunctionFamily`. A family owns a user-facing name such as
  `contains` and its overloads, including logical-domain choices such as list vs. geo.
- Binder: `ExprBinder` or `Binder`, responsible for turning high-level `Expr` into
  `BoundExpr`.
- Low-level implementation trait: keep or evolve `ScalarFnVTable` for bound Vortex
  execution, but consider adding clearer companion traits such as `ScalarFunction`,
  `ScalarKernel`, or `NativeScalarKernel` for the easier implementation APIs.

Working preference unless the code pushes another way: use `FunctionCatalog` for the
registry, `FunctionFamily` for friendly-name overload groups, `Expr` for user-facing
expressions, and `BoundExpr` for the lower-level bound tree. Extension crates or
modules can register multiple families directly with the catalog.

## Expression Tree Shape

Revisit the expression tree instead of only renaming the current struct. The current
representation makes `Root` and `Literal` look like ordinary scalar function calls, even
though they are better understood as expression syntax/IR leaves. A clearer bound tree
may be:

```rust
pub enum BoundExpr {
    Root,
    Literal(Scalar),
    Placeholder(PlaceholderRef),
    Call(BoundCall),
}

pub struct BoundCall {
    function: ScalarFnRef,
    args: Arc<[BoundExpr]>,
    return_dtype: DType,
}
```

`BoundCall` should be the durable output of binding for a callable scalar function. It
should hold the selected low-level function, the already-bound/coerced arguments, and
the resolved return dtype. The actual implementation may also include the selected
overload identity, source spans, display name, nullability, determinism/fallibility
metadata, or other binder diagnostics, but it should not need to recompute
`return_dtype` during execution.

The semantic split should stay clear: root, literals, and placeholders are primitive
expression leaves; scalar functions are `BoundCall` nodes.

Keep `Literal` as a first-class expression leaf rather than a scalar function. A literal
does not consume child arrays and its output length comes from the evaluation scope, so
modeling it as a zero-argument scalar call blurs syntax with callable function
execution. Binding should resolve any untyped or language-level literal into a typed
scalar value; if `Scalar` is not enough to preserve that information, use a small
`BoundLiteral` wrapper with the coerced value, dtype, and optional source/display
metadata. Evaluation lowers the bound literal to a `ConstantArray` for the current
scope length.

Real zero-argument callable functions, if Vortex adds them later, should still be
`BoundCall` nodes. The distinction is semantic: literals are data embedded in the
expression tree; nullary scalar functions are executable functions whose values come
from function semantics.

Do not add a separate `Field` variant to `BoundExpr` if field access is implemented by
the `GetField` / `GetItem` scalar function. Keeping it as a call preserves normal scalar
function dispatch, including specialized kernels, engine registration, and pushdown.
Keep field syntax in the user-facing unbound `Expr` tree. `Expr::Field` is the ergonomic
DSL form for column and nested-field references. Binding should lower it to the
appropriate field-access `BoundCall`.

Include a first-class placeholder variant for values that are bound enough to type-check
but supplied by the execution context. Current examples include `RowIdx` and `RowCount`,
which are modeled as zero-argument scalar functions today but must be supplied or
substituted by scan/pruning execution scope before evaluation. The new expression tree
should not need to pretend those are ordinary scalar function calls.

A possible shape is:

```rust
pub struct PlaceholderRef(Arc<dyn DynPlaceholder>);

pub trait DynPlaceholder: Send + Sync {
    fn id(&self) -> PlaceholderId;
    fn dtype(&self) -> &DType;
    fn payload(&self) -> &(dyn Any + Send + Sync);
}
```

Examples of placeholder IDs include `row_index`, `row_offset`, `row_count`, and
user/query parameters. The payload can carry typed metadata where needed, using the same
typed-trait plus erased-`Dyn*` pattern as extension dtypes and scalar functions.
The erased `Any` payload should be an implementation detail for placeholder metadata,
not the value-delivery API used by expression kernels.

Placeholders are not scalar functions. They are expression leaves whose values come from
the execution context. `ExecutionCtx`, or a scan-specific wrapper around it, should be
able to resolve a placeholder into an explicit value for the current batch/scope:

```rust
pub enum PlaceholderValue {
    Scalar(Scalar),
    Array(ArrayRef),
}
```

A scalar placeholder value can be lifted to a constant array at the evaluation boundary.
If a placeholder is still unresolved when kernels would execute, evaluation should fail
before dispatching into scalar function kernels.

The high-level user-facing `Expr` should likely be a separate unbound tree with similar
syntax-level variants, but with unresolved call names instead of `ScalarFnRef`:

```rust
pub enum Expr {
    Root,
    // User-facing field syntax; lowers to a GetField/GetItem BoundCall.
    Field(FieldPath),
    Literal(Scalar),
    Placeholder(PlaceholderRef),
    Call {
        name: FunctionName,
        args: Vec<Expr>,
    },
}
```

The binder lowers `Expr` to `BoundExpr` by resolving fields, overloads, coercions,
extension dtype semantics, and function options. If `Expr::Field` exists, it is only
user-facing syntax and should lower to `BoundExpr::Call` using the field-access scalar
function.
Prefer the shape that makes traversal, display, serialization, and diagnostics easiest
to understand while keeping execution semantics in scalar functions.

## Trait Design Principles

This work needs a small set of well-named, layered traits. Avoid a few large traits that
mix logical binding, dtype semantics, Vortex execution, and engine registration. Each
trait should answer one question, and adapters should compose those answers into the
larger behavior.

Existing trait names and boundaries are not sacred. It is acceptable to rename, split,
merge, or replace current traits such as `ScalarFnVTable` and `ExtVTable` if that yields
a clearer public model. Use temporary aliases and adapter impls where they reduce
migration risk, but do not preserve an awkward trait shape solely for compatibility on
this branch.

Design principles:

- Keep core traits engine-agnostic. `vortex-array` should not need to know DuckDB,
  DataFusion, Python, Java, or Arrow-specific registration details beyond stable
  interchange hooks.
- Prefer typed implementation traits plus erased registry traits. Implementers should
  get strongly typed APIs; catalogs and sessions should store object-safe erased
  handles.
- Prefer Vortex's existing pattern of non-object-safe typed traits with associated
  types where those associated types make implementations clearer, paired with an
  object-safe erased `Dyn*` trait for storage in registries and sessions. For example,
  a public `Foo` / `FooVTable`-style trait can expose associated `Options`,
  `Metadata`, `NativeValue`, or kernel input/output types, while a sealed `DynFoo`
  bridge handles dynamic dispatch.
- Separate logical semantics from execution kernels. Signatures, coercion, nullability,
  determinism, fallibility, and return types are not the same concern as looping over
  buffers.
- Separate Vortex execution from engine execution. A function may have a generic Vortex
  array kernel, a DuckDB-vector kernel, a DataFusion/Arrow kernel, all of them, or none.
- Make fallback explicit. Adapters should report `native`, `extension`, `storage`, or
  `unsupported` rather than silently degrading a logical type to its storage dtype.
- Keep binding centralized. Overload resolution and coercion should happen in the
  binder/catalog layer, not scattered through language bindings or engine integrations.
- Make the easy path excellent. Adding a scalar function over canonical arrays should
  require implementing a small kernel trait and a small semantic declaration, not a full
  vtable by hand.
- Preserve escape hatches. Low-level `BoundExpr` construction and custom engine kernels
  remain available for integrations that need exact control.

Proposed trait families:

- `ExtVTable` or its successor: logical extension dtype semantics, storage validation,
  metadata, scalar unpacking, coercion, and least-supertype rules.
- `FunctionFamily`: owns a friendly DSL name such as `contains` and its overloads.
- `FunctionOverload`: describes one bindable overload, including argument patterns,
  coercion policy, return type, diagnostics, and the bound function constructor.
- `BoundScalarFn` or evolved `ScalarFnVTable`: the low-level already-bound scalar
  function used by `BoundExpr` and `ScalarFnArray`.
- `ArgumentMatcher`: a broad physical-shape/readiness predicate for each scalar
  argument, likely reusing the existing `DonePredicate` shape. Matchers should say
  "this child is ready enough to build the typed reader" and must not become a second
  dtype system.
- `RowInput` / `RowReader`: typed, non-object-safe helpers for reading canonical and
  constant inputs by row without per-row dynamic dispatch. Readers should be concrete
  enums that can specialize array/constant cases outside the hot loop.
- `RowOutput`: typed output builders for primitive, bool, UTF-8, binary, and scalar
  fallback outputs. The first version can allocate canonical arrays; buffer reuse and
  non-canonical output can come later.
- `UnaryKernel`, `BinaryKernel`, `PairwiseKernel`, or similar: small implementation
  traits lifted by adapters into null handling, constant dispatch, vector loops, and
  array construction.
- `UnaryRowFn`, `BinaryRowFn`, and similar row-oriented helper traits: non-object-safe,
  monomorphized implementation APIs for simple scalar functions. Initial support should
  focus on infallible functions with normal null propagation.
- `EngineTypeAdapter`: maps extension dtypes into a target engine representation,
  including Arrow import/export, engine-native logical types, and storage fallbacks.
- `EngineFunctionAdapter`: registers or maps a function overload into a target engine
  when a compatible type representation, native function, or kernel is available.

Before the full migration, prototype the trait stack with a narrow vertical slice:
one extension dtype with metadata, one friendly overloaded function name, one bound
Vortex implementation, one high-level language binding, and at least one engine adapter.
The result should be judged on whether a new implementer can understand which trait to
implement without reading the entire expression subsystem.

## Required Architecture Work

- Define the API boundary between high-level unbound expressions and low-level bound
  expressions.
- Rename and restructure the current `Expression` type toward `BoundExpr`, including a
  possible enum representation, without over-optimizing for long-term compatibility.
- Define the high-level expression AST, including literals, fields, function calls,
  aggregates, aliases, projection expressions, and filter predicates.
- Build a catalog and binder that resolve friendly names, namespaces, overloads,
  argument types, coercions, nullability, and return types.
- Decide how coercion rules are represented: per function, per overload, per catalog,
  or a combination.
- Define diagnostics for failed binding so language APIs can return useful errors rather
  than low-level `return_dtype` failures.
- Decide how aggregate functions fit alongside scalar functions, including signatures,
  state, null handling, and engine registration.
- Define how extension dtype metadata, coercion, and external representations participate
  in function binding.
- Define the row-helper adapter boundary for scalar functions: which logic is semantic
  metadata, which logic is typed row compute, and which logic belongs to physical
  parent-kernel dispatch.
- Define scalar-function benchmark coverage before porting many functions, so API and
  execution changes can be evaluated with real measurements.
- Preserve explicit lower-level construction for integrations that already know the
  exact scalar function they want.
- Define serialization and compatibility expectations for the bound representation,
  including any changes needed in `vortex-proto`.

## Function Implementation Model

Implementing scalar functions over canonical array types should become much easier than
hand-writing all dispatch behavior.

The target shape is a set of traits and adapters where an implementer can write the
smallest useful kernel, for example a unary or binary pairwise function over native Rust
values, and Vortex lifts it into the full execution surface:

- constant-array dispatch;
- nullable input and validity handling;
- scalar/scalar, scalar/array, and array/array cases;
- canonical-array downcasting and validation;
- output array construction;
- fallibility and error propagation;
- auto-vectorization or tight loops where possible;
- preservation of opportunities for deferred `ScalarFnArray` execution and parent
  pushdown.

This likely means separating semantic function metadata from execution kernels more
clearly than today. The existing `ScalarFnVTable` already carries several semantic hooks
such as arity, coercion, return dtype, null sensitivity, fallibility, simplification,
statistics rewrite, and execution. The new work should decide which hooks stay on the
low-level bound function trait and which move into catalog/binder-facing function
definitions.

The scalar row-helper plan is a good first implementation model for ordinary scalar
functions. The key idea is to let simple functions be implemented as typed Rust row
logic while adapters lift that logic into Vortex arrays. The helper traits do not need
to be object-safe because the existing scalar function path already uses non-object-safe
typed traits and erases them later through typed instances and `Dyn*` handles.

Initial row-helper APIs should be deliberately small:

- `RowInput` supplies an argument matcher and constructs a typed reader after the child
  is physically ready.
- `RowReader` exposes `len`, null checks, and borrowed per-row values. Concrete reader
  enums should cover canonical array and constant cases without a dyn call in the hot
  loop.
- `RowOutput` owns dtype calculation, builder construction, pushing values/nulls, and
  finishing an output array.
- `UnaryRowFn` and `BinaryRowFn` expose the actual row logic, with declared null
  behavior and fallibility.

The first supported mode should be normal null propagation plus infallible functions.
`not`, primitive comparisons, simple arithmetic, and simple infallible casts are good
first candidates. `is_null` and `is_not_null` become good candidates once the helper
layer supports null-observing functions. Delay `AND`, `OR`, `CASE`, `TRY`, fallible
casts, division, parsing, bitmap word-at-a-time kernels, and functions that should
preserve a special encoding until the execution model can represent control flow, row
errors, and specialized vector algorithms correctly.

Row helpers are not meant to replace every optimized kernel. They are the default easy
path for canonical/constant scalar execution, and specialized kernels should still
bypass them when they can preserve encodings, exploit bitmaps, or use a better
vectorized algorithm.

## Scalar Execution Model

Changes to the scalar execution scheduler can come after the initial DSL and row-helper
work, but they should remain part of the design target.

The long-term direction is to make `ScalarFnArray` a scheduler for callable scalar
functions rather than the place that always falls back to `ScalarFnVTable::execute`.
The existing executor already has parent-kernel hooks and `DonePredicate` readiness
predicates. A later migration should build on that:

- Add argument matchers to the bound scalar function trait, defaulting to "any array".
- Teach `ScalarFnArray` to drive each child only until that child's matcher is
  satisfied, rather than always canonicalizing or forcing full execution.
- Register row-helper adapters as canonical/constant `execute_parent` fallback kernels.
- Keep the legacy scalar `execute` path as a temporary compatibility fallback while
  coverage is incomplete.
- Eventually make a ready `ScalarFnArray` with no matching parent kernel a clear
  "missing physical kernel" error instead of silently doing an ad hoc fallback.

Multi-argument dispatch should be deterministic. Prefer dispatching from the first
non-constant child with a matching parent kernel; otherwise dispatch from child 0. The
kernel must inspect siblings and decline if their physical shape does not satisfy the
function's argument matchers. All-constant inputs can be evaluated once into a constant
array when the function is deterministic and infallible for those inputs.

Selection masks, demand masks, and vectorized row errors should be designed into the
helper APIs but not forced into the first implementation. The first row-helper adapter
can run with `selection = AllTrue`, `demand = AllTrue`, and normal error propagation.
Later work should add:

- selected execution over `Mask` without evaluating irrelevant rows;
- demand-aware output so callers can avoid computing unused rows or fields;
- vectorized row errors / execution error masks for checked arithmetic, casts, parsing,
  division, and `TRY`-style behavior;
- loop specialization outside the hot loop for reader shape, mask shape, nullability,
  output type, and fallibility mode.

This migration also creates an opportunity to improve crate boundaries. `vortex-array`
should keep scalar function metadata, expression scaffolding, `ScalarFnArray`,
matchers, and parent-kernel registration hooks. Heavier compute families can move to
smaller crates or feature-gated modules over time: primitive arithmetic/comparison,
boolean kernels, cast kernels, string predicates, fill/null kernels, list/variant
kernels, and domain-specific kernels. Standard sessions can register the full built-in
compute surface by default; minimal sessions can register less and report missing
physical kernels clearly.

`Binary` should be split before broad physical kernel registration. A single
`Binary + Operator` ID makes kernels inspect an operator value after matching one
function ID. Splitting comparison, arithmetic, and boolean families gives cleaner
function IDs, signatures, fallibility, coercion, and engine/kernel registration.

## Scalar Function Benchmarking

Add a focused benchmarking harness for scalar functions before broad row-helper ports.
The repository already uses Divan benches under `vortex-array/benches`, so the first
version should fit that model rather than inventing a separate benchmark stack.

The harness should make it easy to benchmark one scalar function across a matrix of
physical input shapes and data distributions:

- array/array, array/constant, constant/array, and constant/constant inputs;
- primitive, bool, UTF-8, binary, list, and extension dtype cases as they become
  relevant;
- nullable and non-nullable inputs, with configurable null density and clustering;
- small, medium, and large batch lengths;
- canonical inputs first, then compressed/dictionary/chunked inputs where parent
  kernels matter;
- eager execution vs. deferred `ScalarFnArray` materialization, measured separately
  from expression construction unless the benchmark explicitly targets binding or
  planning overhead.

Benchmark helpers should consume results enough to prevent dead-code elimination and
should report per-value throughput alongside wall-clock timing where practical. The
same harness should support old handwritten kernels and row-helper adapters so ports
can prove that the easy path is not a systematic regression.

## Engine Integration Model

The function traits should expose enough metadata and implementation hooks to register
Vortex functions and types into external engines, especially Arrow, DuckDB, and
DataFusion.

Treat Arrow as an engine in this design. It is not a query engine, but it has its own
type model, extension metadata conventions, scalar/array representation choices, and
import/export conversion hooks. Calling it an engine keeps the integration surface broad
enough to cover both execution engines such as DuckDB/DataFusion and representation
engines such as Arrow.

The desired outcome is not only conversion from external engine expressions into Vortex
`BoundExpr`. For execution engines, Vortex should also be able to register efficient
native kernels where supported:

- DataFusion should be able to register Vortex-backed scalar and aggregate functions
  using the same catalog metadata and coercion/signature information.
- DuckDB should be able to call efficient kernels over DuckDB vectors when possible,
  avoiding prohibitively expensive DuckDB-to-Vortex-to-DuckDB conversion for tight
  scalar execution paths.
- Arrow adapters should define import/export behavior, extension metadata, native Arrow
  type choices, and fallback-to-storage behavior using the same extension dtype policy
  as execution engines.
- When an engine already has a mature native type and function ecosystem for a common
  logical domain, prefer mapping into that ecosystem over registering parallel
  Vortex-specific types or functions. For example, DuckDB geospatial integration should
  use DuckDB's spatial extension types and functions where the semantics match.
- Engine adapters should be generated or implemented from the same function definition
  where practical, so adding a function does not require duplicating semantics in every
  integration.
- The API should still allow engine-specific overrides when the native vector format can
  be exploited more efficiently than the generic Vortex array path.

## Extension DTypes and Type Interop

Extension dtypes are central to the DSL design, not a conversion afterthought. Vortex
logical types can be richer than any one execution engine's type system. Each extension
dtype should declare both its Vortex semantics and its available external
representations.

Existing Vortex extension dtypes already have the right core shape: an extension ID,
metadata, a storage dtype, scalar validation, native-value unpacking, and dtype coercion
hooks. The new catalog/binder layer should compose these dtype hooks with function
overload resolution.

Do not put every engine's type-registration details directly on the core extension dtype
trait. The core dtype trait should stay engine-agnostic and describe logical semantics:
identity, metadata, storage, validation, coercion, least-supertypes, and any stable
interchange representation. Engine-specific details should live in adapter plugins keyed
by extension ID. Those plugins can use the shared semantic core where possible, but they
own engine-specific facts such as DuckDB extension availability, DataFusion/Arrow type
constructors, native vector layout, fallback policy, and function registration.

In other words, prefer a hybrid model: extension dtypes expose enough common information
for correct binding and safe fallback decisions, while DuckDB/DataFusion/Arrow/language
adapters register the concrete representation each engine needs. This avoids bloating
`ExtVTable` with unstable per-engine APIs while still preventing ad hoc integration code
from silently erasing logical type semantics.

Engine interop needs an explicit policy per extension type:

- **Native mirror:** the engine has a first-class equivalent. Examples include temporal
  types, DuckDB geometry via its spatial extension, UUID, JSON, and Decimal where
  supported. In this case Vortex should use the engine-native logical type and should
  prefer the engine's existing function family when the semantics match. For example,
  a Vortex geospatial type should map to DuckDB spatial's geometry representation and
  compatible `ST_*` operations rather than registering a separate Vortex geospatial
  function namespace in DuckDB. Register Vortex-specific functions only for semantics
  the engine does not already provide, or when a native function is observably
  incompatible.
- **Standard extension mirror:** the engine or interchange layer supports a standard
  extension representation, especially Arrow extension metadata. UUID and geo/WKB are
  examples of types that can preserve logical identity while using a storage type.
- **Storage fallback:** the engine does not understand the logical type, but can carry
  the storage dtype. For example, `vortex.img.jpeg` might be stored as binary bytes. The
  engine may scan, project, compare for equality, or pass values through, but it should
  not claim JPEG-aware semantics unless the relevant function family is registered.
- **Opaque / Vortex-only:** the engine cannot safely represent the type or the operation.
  In that case binding or pushdown should fail with a useful diagnostic, or the query
  should stay inside Vortex execution.

The binder should resolve functions against logical extension dtypes, not only their
storage dtypes. For example, `contains(a, b)` should be able to choose `geo.contains`,
`list.contains`, or another overload based on the bound argument dtypes. Storage fallback
is a representation choice for interop; it must not erase logical semantics inside the
Vortex DSL.

This suggests adding an engine adapter layer that generalizes the existing Arrow
extension import/export plugin model:

- extension dtype registration remains session-scoped;
- each extension can optionally expose Arrow, DuckDB, DataFusion, and language-binding
  representations;
- engine adapters can ask whether a type has a native representation, a storage
  fallback, or no safe representation;
- function registration can declare which engine representations it supports, including
  engine-native functions, native vector kernels, Vortex kernels, and fallback modes;
- high-level language APIs should expose logical Vortex dtypes, even when their host
  language or engine sees a native mirror or storage fallback underneath.

Good first plugin extraction candidates:

- `vortex-temporal`: move date, time, and timestamp extension dtype registration and
  temporal-specific function families out of core once the adapter shape is ready. This
  proves common native mirrors across Arrow, DuckDB, DataFusion, and language APIs,
  including units and timezone metadata.
- `vortex-uuid`: extract UUID extension support and parse/format/cast helpers. This is
  the smallest native-mirror plugin and should prefer engine-native UUID support where
  available.
- `vortex-url`: add URL, hostname/domain, and IP address logical types. Storage can be
  string or binary depending on the subtype, while adapters map to native engine support
  when available and otherwise expose validation, parsing, normalization, containment,
  and component-extraction functions through Vortex.

For `vortex.img.jpeg`, the likely first version is a Vortex extension dtype with binary
storage, an optional Arrow extension representation, no assumed DuckDB/DataFusion native
logical mirror, and explicit image function families for operations such as metadata
extraction, decoding, resizing, embedding, or similarity. Engines should not silently
treat it as ordinary binary for functions that depend on JPEG semantics.

## Roadmap

Start with the low-level bound representation. The existing `Expression` type is already
the execution-facing IR, and replacing it with a real `BoundExpr` is a large, early
migration that everything else will build on. Do not start by introducing the high-level
DSL or catalog API; that can wait until the lower-level model is stable and we are ready
to expose nicer PyVortex and language-binding APIs.

### Phase 1: BoundExpr Foundation

- Replace today's `Expression` struct with `BoundExpr` as the lower-level, already-bound
  expression representation.
- Prefer the final enum shape early: `Root`, `Literal`, `Placeholder`, and
  `Call(BoundCall)`.
- Introduce `BoundCall` with the selected `ScalarFnRef`, bound/coerced arguments, and
  resolved `return_dtype`.
- Keep `Field` out of `BoundExpr`; field access should lower to a field-access
  `BoundCall`.
- Move `Root`, `Literal`, `RowIdx`, and `RowCount`-style behavior out of the
  "pretend scalar function" model where practical.
- Update traversal, formatting, proto/serialization, stats rewrites, pruning, scan
  construction, tests, and direct expression builders to consume `BoundExpr`.
- Keep execution on the existing `ScalarFnVTable` / `ScalarFnArray` path initially. The
  first milestone should prove the bound representation, not a new execution engine or
  a friendly high-level DSL.

Exit criterion: existing expression construction, scan integration, stats/pruning
rewrites, serialization tests, and execution tests operate on `BoundExpr`, with
`ScanBuilder` accepting the low-level bound representation.

### Phase 2: Function Implementation Adapters

- Before finalizing the row-helper trait design, inspect prior art in
  `/Users/ngates/git/velox` and `/Users/ngates/git/duckdb`.
- From Velox, look at `velox/type/SimpleFunctionApi.h`,
  `velox/expression/SimpleFunctionAdapter.h`, and
  `velox/expression/VectorFunction.h` for typed simple-function APIs, adapter lifting
  into vector execution, selectivity-vector handling, flat/no-null fast paths, constant
  input treatment, and memory/string-encoding reuse hooks.
- From DuckDB, look at `src/include/duckdb/function/function.hpp` and
  `src/include/duckdb/function/scalar_function.hpp` for bound function data,
  signatures, return types, null handling, stability/fallibility properties, bind
  callbacks, local state, statistics propagation, direct selection callbacks, and
  serialization hooks.
- Use those systems as design inspiration, not as APIs to copy. The Vortex shape should
  still be Rust-native, typed where useful, and compatible with the existing
  `ScalarFnVTable` plus erased `Dyn*` pattern.
- Design the first row-helper traits: `RowInput`, concrete `RowReader` enums,
  `RowOutput`, `UnaryRowFn`, and `BinaryRowFn`.
- Add the scalar-function benchmark harness before porting many functions, so the old
  and new paths can be compared function-by-function.
- Lift one simple native Rust implementation into Vortex array execution through the
  helper layer, including constant dispatch, null handling, scalar/array combinations,
  output construction, and fallibility metadata.
- Decide which semantic hooks stay on the bound scalar function trait and which belong
  on `FunctionOverload`.
- Keep old handwritten scalar functions working while the adapter model proves itself.
- Port a small first set of functions such as `not`, primitive comparisons, and simple
  arithmetic. Add `is_null` / `is_not_null` only after null-observing helper behavior is
  explicit. Avoid control-flow functions and fallible row-error functions in this phase.

Exit criterion: a new scalar function can be implemented without hand-writing all
canonical array, constant, and null dispatch logic; the scalar benchmark harness can
compare it against the previous implementation shape; and the Phase 2 design notes
record which Velox/DuckDB ideas were adopted or rejected.

### Phase 3: Scalar Execution Scheduler Migration

- Add argument matchers to bound scalar functions, defaulting to the existing "any array"
  readiness behavior.
- Teach `ScalarFnArray` to use argument matchers for iterative child execution.
- Register row-helper adapters as canonical/constant `execute_parent` fallback kernels
  for the scalar functions they support.
- Keep the legacy scalar `execute` path as a temporary compatibility fallback until
  enough functions have physical kernels.
- Add selected execution, vectorized row errors, and demand-aware loops in that order,
  after the all-true-selection/all-demanded path is correct.
- Split overloaded scalar families where needed, starting with `Binary` into comparison,
  arithmetic, and boolean families.

Exit criterion: a row-helper-backed scalar function can execute through parent-kernel
dispatch over canonical and constant inputs, with benchmark coverage showing the cost of
the new scheduler path.

### Phase 4: Extension DType Representation Policy

- Add the first engine representation model for extension dtypes: native mirror,
  standard extension mirror, storage fallback, or unsupported.
- For common native mirrors, define whether Vortex maps to the engine's existing type
  and function family instead of registering Vortex-specific functions. Geospatial on
  DuckDB should use DuckDB spatial where semantics match; UUID, JSON, temporal, and
  Decimal should similarly prefer native engine support when available.
- Plan extraction of `vortex-temporal`, `vortex-uuid`, and `vortex-url` as early plugin
  crates. `vortex-url` should include URL, domain/host, and IP address logical types.
- Wire this into binding diagnostics so logical extension types are not silently erased
  to storage types.
- Prove the model with an existing extension dtype such as UUID or WKB, and optionally a
  deliberately non-native example shaped like `vortex.img.jpeg`.

Exit criterion: the binder and engine adapters can ask what an extension dtype means
logically and how, or whether, it can be represented outside Vortex.

### Phase 5: Catalog, Binder, and Friendly DSL

- Add the high-level unbound `Expr` tree with user-facing syntax such as `Expr::Field`,
  literals, placeholders, and unresolved named calls.
- Add the first versions of `FunctionCatalog`, `FunctionFamily`, `FunctionOverload`,
  `BindCtx`, and `BindError`.
- Use the typed-trait plus erased-`Dyn*` pattern for catalog storage where it helps keep
  implementation APIs strongly typed.
- Bind one friendly DSL function name end-to-end. `contains` is the preferred test case
  because it naturally demonstrates overload resolution between logical domains such as
  lists and geo values.
- Keep the public surface modest until the Python/Java/CXX APIs are ready; the initial
  Rust binder can be internal or experimental if that reduces churn.

Exit criterion: a Rust test can construct high-level `Expr::call("contains", ...)`,
bind it against input dtypes through a `FunctionCatalog`, and receive a `BoundExpr`
whose selected low-level function and return dtype are deterministic and diagnosable.

### Phase 6: First Engine Adapters

- Build the Arrow and DuckDB adapter paths in parallel using the new type/function
  metadata. Arrow exercises import/export, extension metadata, and storage fallback;
  DuckDB keeps the design honest for query-engine native types, native functions, and
  vector kernels.
- DataFusion can follow using Arrow-facing metadata plus function catalog metadata.

Exit criterion: one function/type pair can be represented in Arrow and mapped into
DuckDB from the same semantic definitions used by the Vortex binder.

### Phase 7: Language API Surface

- Expose the high-level DSL in Python first, with Java/CXX/FFI following once the Rust
  API stabilizes.
- Keep lower-level `BoundExpr` construction available for advanced users and engine
  integrations.
- Make binding errors user-facing and specific.

Exit criterion: a Python user can build a friendly expression, bind it through the
catalog, and pass the resulting `BoundExpr` into scan construction.

### Phase 8: Migration and Cleanup

- Migrate existing direct expression construction to the new names and layers.
- Move function-specific coercion and overload logic out of scattered integrations and
  into the catalog/binder model.
- Consider moving heavier compute families out of the `vortex-array` hot compile path
  once registration boundaries are clear.
- Split compatibility aliases and shims into separate cleanup PRs where possible.
- Update docs, examples, and integration tests around the final public API names.

Exit criterion: `ScanBuilder` and integrations consume `BoundExpr`; high-level APIs use
the DSL/catalog path; compatibility shims are either removed or clearly documented.

## Commit Strategy

The commit log should be useful, but it is not the source of truth for the eventual PR
split. During API churn, avoid spending time polishing history that is likely to be
invalidated by the next design turn.

When practical:

- Make commits at coherent checkpoints.
- Keep unrelated subsystems separate when the split is cheap.
- Use clear prefixes such as `expr:`, `scan:`, `layout:`, `file:`, `datafusion:`,
  `duckdb:`, `python:`, `ffi:`, `docs:`, `test:`, `bench:`, `fix:`, and `plan:`.
- Prefer new fixup/checkpoint commits over amending already-shared work.
- Separate large generated-code or mechanical churn from semantic changes when possible.
- Include required sign-offs on commits created by agents:

```text
Signed-off-by: "COMMITTER" <COMMITTER_EMAIL>
```

It is fine if the branch history becomes imperfect. The later split will use the commit
log as evidence, not as a binding review structure.

## Live Tracking

Keep lightweight notes here or in follow-up design docs as the branch evolves:

- Major decisions and reversals.
- Stable checkpoints and the checks that passed there.
- Known temporary breakages.
- Areas that should become separate PRs.
- Deferred work that should not block the first review series.

Avoid turning this file into a detailed task tracker. It should stay small enough that a
future reviewer or agent can quickly recover the intent of the branch.

### 2026-06-12: Phase 1 checkpoint — BoundExpr lands in vortex-array

- `Expression` is replaced by the `BoundExpr` enum: `Root(DType)`, `Literal(Scalar)`,
  `Placeholder(PlaceholderRef)`, `Call(BoundCall)`. Decision: **Root carries its bound
  scope dtype**, so every node's dtype is self-contained — `BoundExpr::dtype()` takes
  no scope and `BoundCall` stores `return_dtype` resolved at construction, which is now
  fallible. Scope params dropped from `simplify`/`optimize*`/`falsify`/`satisfy`.
- Root/Literal/RowCount are no longer "pretend scalar functions"; RowCount is the first
  placeholder (RowIdx follows with vortex-layout). Placeholders survive `apply()` via
  the internal `PlaceholderFn` marker so array-level substitution keeps working.
- Proto keeps the `vortex.root`/`vortex.literal` wire ids; Root metadata now stores the
  scope dtype. Wire change: legacy empty-metadata Roots error on read (pb::Expr is not
  embedded in the file format; in-repo consumers are round-trip tests only).
- `coerce_expression` deleted — ill-typed trees are unconstructible under bound typing;
  coercion centralizes in the Phase 5 binder. `checked_pruning_expr` now takes the data
  scope and synthesizes a deterministic stats-scope struct for bound stat references.
- Known temporary breakage: all downstream crates (vortex-layout/-scan/-file, engines,
  bindings) do not compile until the next checkpoints migrate them.
- Checks passed here: `cargo build/nextest/test --doc/clippy --all-targets` for
  `vortex-array`, `--features arbitrary` check, `-D warnings` release check, fmt.
- Deferred: `ReduceCtx` has no literal-node constructor, so `GetItem::reduce` skips the
  nullable-pack rewrite (expression-domain `simplify_untyped` still covers it; TODO in
  code). Placeholder serde and `PlaceholderValue`/ExecutionCtx resolution deferred.

## Tentative PR Split Areas

The final PR boundaries will be chosen from the completed diff, not guessed up front.
Likely review strata include:

- Public API layering: high-level DSL, low-level `BoundExpr`, and migration aliases.
- Function catalog, function families, overload binding, diagnostics, and coercion rules.
- Extension dtype catalog integration and engine representation policies.
- Core expression model, scalar functions, aggregate functions, literals, and field
  access.
- Scalar row-helper traits and adapters for canonical/constant arrays.
- Scalar-function benchmark harness and baseline benchmark cases.
- Scalar execution scheduler changes, argument matchers, parent-kernel fallback, and
  later selection/demand/error-mask support.
- Optional compute crate or feature-boundary cleanup once kernel registration is clear.
- Scan, layout, file, and predicate/projection pushdown integration.
- DataFusion and DuckDB conversion, pushdown, registration, and native-kernel adapters.
- Python, CXX, FFI, and JNI binding updates.
- Documentation, examples, and migration notes.
- Cleanup of obsolete expression paths and compatibility shims.

These are candidates, not commitments. If the implementation discovers a better split,
prefer the split that gives reviewers independently understandable, testable diffs.

## Later Split Protocol

When the implementation is ready to split:

1. Start from fresh `develop`, not from assumptions about the old base.
2. Inventory the integration branch with `git log --first-parent`, `git diff --stat`,
   `git diff --dirstat`, and targeted code inspection.
3. Propose a PR series before creating branches. Each PR should have a clear purpose,
   a bounded diff, and a test plan.
4. Use cherry-picks when the history already matches the desired boundary.
5. Reconstruct patches manually when commits are tangled; reviewability matters more
   than preserving the original exploratory commits.
6. Keep each split PR independently buildable when possible. If a PR must be staged
   behind another, make that dependency explicit.
7. Run narrow tests for the crates or bindings touched by each PR, then broader checks
   at integration points.
8. Do not include exploratory scaffolding or this process plan in split PRs unless it is
   still useful to reviewers.

## Verification Expectations

Use narrow checks while iterating and broader checks at stable checkpoints.

- Rust crate changes: `cargo build -p <crate>` and targeted `cargo nextest run -p <crate>`.
- Cross-crate public API changes: broaden to affected dependent crates.
- Scalar benchmark harness changes: run the targeted Divan bench, for example
  `cargo bench -p vortex-array --bench <bench-name>`, and record the compared scenarios
  in the checkpoint notes or PR body.
- Rust formatting and linting before handoff of behavior changes:
  `cargo +nightly fmt --all` and `cargo clippy --all-targets --all-features`.
- Python binding changes: use the targeted `py_compile`, `pytest`, `ruff`, and
  `basedpyright` commands described in `AGENTS.md`.
- Documentation changes: run the relevant doctests or docs build when the docs behavior
  changed.

Record notable checkpoint commands and failures in this file or in the PR body when the
branch is split.

## Agent Guidance

Agents working on this branch should:

- Read `AGENTS.md` and this file before making changes.
- Preserve user work and avoid destructive git operations unless explicitly requested.
- Keep changes scoped to the current task, but do not pretend the broader branch is
  linear or clean.
- Update this plan when the branch process changes or when new PR boundaries become
  obvious.
- When asked to split the branch, first produce a concrete split proposal and test plan,
  then create branches/PRs only after the split is agreed or explicitly requested.
