# Expressions

Expressions represent computations over arrays. They form a tree where each node is either
a leaf (field reference, literal, placeholder) or an operation applied to child expressions.

## Current Design

Today `Expression` is a flat struct: every node holds a `ScalarFn` (vtable + options) and
a list of children. Structural nodes like `Root`, `GetItem`, `Literal`, and `Select` are
all modeled as scalar functions with vtables, even though they don't perform element-wise
computation. This conflates navigation with computation and makes the expression tree
heavier than it needs to be.

## Target Design

`Expression` becomes an enum with four variants:

```rust
enum Expression {
    /// A column/field path, e.g. $.foo.bar
    FieldPath(FieldPath),

    /// A constant value.
    Literal(Scalar),

    /// A scalar function applied to child expressions.
    ScalarFn(ScalarFnRef, Arc<[Expression]>),

    /// A typed placeholder for values injected by the scan/layout layer.
    Placeholder(Arc<dyn Placeholder>),
}
```

### FieldPath

Collapses the current `Root` -> `GetItem("foo")` -> `GetItem("bar")` chains into a single
`FieldPath(["foo", "bar"])`. This is simpler, serializes directly, and does not require a
vtable. `FieldPath` is a concrete type (e.g. a newtype over `Arc<[FieldName]>`).

### Literal

A constant scalar value. No children, no vtable. Pulling it out of `ScalarFn` removes the
degenerate arity-0 "function that ignores all inputs" case.

### ScalarFn

All genuine element-wise operations live here. `ScalarFnRef` is the type-erased vtable
reference following the standard vtable pattern (`ExprVTable` / `Expr<V>` / `ExprRef`).

Current scalar functions:

| Function          | Options              | Arity | Notes                                 |
|-------------------|----------------------|-------|---------------------------------------|
| Binary            | Operator             | 2     | Arithmetic, comparison, boolean logic |
| Cast              | DType                | 1     | Type conversion                       |
| Not               | (none)               | 1     | Boolean negation                      |
| IsNull            | (none)               | 1     | Null test, returns bool               |
| FillNull          | (none)               | 2     | Null replacement                      |
| Mask              | (none)               | 2     | Validity intersection                 |
| Zip               | (none)               | 3     | Conditional / ternary select          |
| Between           | BetweenOptions       | 3     | Range test                            |
| Like              | LikeOptions          | 2     | SQL pattern matching                  |
| ListContains      | (none)               | 2     | List membership test                  |
| DynamicComparison | DynamicComparisonExpr| 1     | Runtime-adjustable filter             |
| Pack              | PackOptions          | N     | Struct assembly from fields           |
| Merge             | DuplicateHandling    | N     | Struct union / field merge            |

### Placeholder

A typed, opaque value injected by the scan or layout layer. `Placeholder` is a trait
(not a string) providing at minimum `id()` and `dtype()`. Examples:

- `RowIdx` — the row index within a layout partition
- Future: partition index, file path, or other scan-provided context

Using `Arc<dyn Placeholder>` rather than a string ensures exhaustive handling and
type-safe downcasting.

### What about Select?

`Select` (field projection with include/exclude modes) is a schema operation, not an
expression. It belongs in the scan/plan layer. If needed temporarily in the expression
tree during migration, it can live as a `ScalarFn`, but the target is to move it out.
