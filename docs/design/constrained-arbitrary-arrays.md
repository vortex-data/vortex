# Design: Composable Constrained Arbitrary Array Generation

## Summary

This document proposes a system for generating arbitrary arrays with structural constraints
in a composable way. This enables fuzzing of complex array encodings (like RunEnd, Sparse, Dict)
where child arrays must satisfy specific invariants (sorted, bounded, etc.) while still allowing
those child arrays to be compressed.

## Motivation

Many Vortex array encodings have structural constraints on their children:

- **RunEnd**: `ends` must be strictly sorted unsigned integers
- **Sparse**: `indices` must be strictly sorted and bounded by array length
- **Dict**: `codes` must be bounded by `values.len()`
- **VarBin/List**: `offsets` must be monotonically increasing, starting at 0

Currently, arbitrary generation for these arrays produces primitive arrays for constrained
children. For example, `ArbitraryRunEndArray` generates `ends` as a plain `PrimitiveArray`.

This limits fuzz coverage because:
1. We don't test compressed representations of constrained arrays
2. We miss bugs in how encodings handle compressed children
3. Real-world data often has compressed sorted/bounded arrays (delta-encoded timestamps, etc.)

## Goals

1. **Composability**: Constrained arrays can themselves be compressed (Delta, FOR, BitPacked, etc.)
2. **Extensibility**: New encodings can declare their constraint capabilities
3. **Reusability**: Common constraint patterns (sorted, bounded) are defined once
4. **Fuzz coverage**: Exercise more code paths by varying child encodings

## Constraint Taxonomy

Based on analysis of all Vortex array encodings, we identify these constraint categories:

### Ordering Constraints

| Constraint | Description | Used By |
|------------|-------------|---------|
| `StrictlySorted` | Each value > previous (no duplicates) | RunEnd ends, Sparse indices |
| `Sorted` | Each value >= previous | VarBin offsets, List offsets |
| `StartsAtZero` | First element must be 0 | VarBin offsets, List offsets |

### Bound Constraints

| Constraint | Description | Used By |
|------------|-------------|---------|
| `UpperBound(n)` | All values < n | Dict codes, Sparse indices, VarBin last offset |
| `LowerBound(n)` | All values >= n | Positive increments |
| `BitWidth(n)` | Values fit in n bits | BitPacked |
| `TargetMax(n)` | Soft target for maximum value | RunEnd ends (target = array length) |

### Type Constraints

| Constraint | Description | Used By |
|------------|-------------|---------|
| `Unsigned` | Must be unsigned integer type | RunEnd ends, Dict codes |
| `IntegerOnly` | Must be integer (not float) | Sequence, BitPacked |
| `AllowedPTypes(list)` | Must be one of specific ptypes | ALP, PCO |

### Nullability Constraints

| Constraint | Description | Used By |
|------------|-------------|---------|
| `NonNullable` | Cannot contain null values | RunEnd ends, offsets arrays |
| `AllValid` | Child must have no actual nulls | MaskedArray child |

### Content Constraints

| Constraint | Description | Used By |
|------------|-------------|---------|
| `ValidUtf8` | Bytes must be valid UTF-8 | VarBin/VarBinView with Utf8 dtype |

## Proposed Design

### Constraint Types

```rust
/// Ordering constraints for array values
#[derive(Clone, Debug, Default)]
pub struct OrderingConstraint {
    /// Values must be strictly increasing (each > previous)
    pub strictly_sorted: bool,
    /// Values must be monotonically increasing (each >= previous)
    pub sorted: bool,
    /// First value must equal this
    pub starts_at: Option<u64>,
}

/// Value range constraints
#[derive(Clone, Debug, Default)]
pub struct BoundConstraint {
    /// All values must be < upper_bound
    pub upper_bound: Option<u64>,
    /// All values must be >= lower_bound
    pub lower_bound: Option<u64>,
    /// Soft target for maximum value (used for sorted arrays)
    pub target_max: Option<u64>,
    /// Values must fit in this many bits
    pub bit_width: Option<u8>,
}

/// Type constraints
#[derive(Clone, Debug, Default)]
pub struct TypeConstraint {
    /// Must be unsigned integer
    pub unsigned: bool,
    /// Must be integer (not float)
    pub integer_only: bool,
    /// If Some, must be one of these ptypes
    pub allowed_ptypes: Option<Vec<PType>>,
}

/// Combined constraints for arbitrary array generation
#[derive(Clone, Debug, Default)]
pub struct ArrayConstraints {
    pub ordering: OrderingConstraint,
    pub bounds: BoundConstraint,
    pub type_constraint: TypeConstraint,
    /// Must be non-nullable
    pub non_nullable: bool,
    /// Must be valid UTF-8 (for binary data)
    pub valid_utf8: bool,
}
```

### Capability Advertisement

Each encoding declares two types of capabilities:

1. **Can Generate**: Constraints the encoding can satisfy when generating from scratch
2. **Preserves**: Constraints preserved when wrapping another array

```rust
/// Trait for encodings that support constrained arbitrary generation
pub trait ArbitraryConstrained {
    /// Constraints this encoding can generate directly (leaf capability)
    fn can_generate() -> &'static [ConstraintKind];

    /// Constraints this encoding preserves when wrapping a child
    /// (if child satisfies constraint, so does the wrapped array)
    fn preserves() -> &'static [ConstraintKind];

    /// Generate an arbitrary array satisfying the given constraints
    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> arbitrary::Result<ArrayRef>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintKind {
    StrictlySorted,
    Sorted,
    StartsAtZero,
    BoundedAbove,
    BoundedBelow,
    BitWidthBounded,
    NonNullable,
    Unsigned,
    IntegerOnly,
}
```

### Encoding Capabilities Matrix

| Encoding | Can Generate | Preserves |
|----------|--------------|-----------|
| **Primitive** | All (base case) | N/A (leaf) |
| **Constant** | All (trivially) | N/A (leaf) |
| **Sequence** | StrictlySorted*, Sorted*, NonNullable | N/A (leaf) |
| **Delta** | - | StrictlySorted*, Sorted*, BoundedAbove |
| **FOR** | - | StrictlySorted, Sorted, BoundedAbove |
| **BitPacked** | - | StrictlySorted, Sorted, BitWidthBounded |
| **ZigZag** | - | (none - transforms value space) |
| **Dict** | - | (none - reorders by code) |
| **RunEnd** | - | (none - different structure) |
| **Sparse** | - | StrictlySorted*, Sorted* (for patch values) |

*Conditional on parameters (e.g., Sequence is sorted only if multiplier >= 0)

### Generation Strategy

For sorted arrays, we use the **increments approach**:

1. Strictly sorted = cumulative sum of positive increments
2. Sorted = cumulative sum of non-negative increments

This is maximally composable because the **increments themselves can be any encoding**:

```rust
fn arbitrary_strictly_sorted(
    u: &mut Unstructured,
    len: usize,
    dtype: &DType,
    target_max: Option<u64>,
) -> Result<ArrayRef> {
    // Generate positive increments (each >= 1)
    // These increments CAN BE COMPRESSED (Constant, RunEnd, BitPacked, etc.)
    let increment_constraints = ArrayConstraints {
        bounds: BoundConstraint { lower_bound: Some(1), ..default() },
        non_nullable: true,
        ..default()
    };

    let increments = arbitrary_constrained_array(
        u, Some(len), dtype, &increment_constraints
    )?;

    // Option A: Wrap in Delta encoding (stores increments, reconstructs on access)
    // Option B: Materialize cumsum as primitive, then optionally compress

    if u.arbitrary()? {
        // Return as Delta-encoded (increments stored directly)
        Ok(DeltaArray::from_increments(increments)?.into_array())
    } else {
        // Materialize and optionally apply preserving compression
        let materialized = materialize_cumsum(increments)?;
        maybe_compress_sorted(u, materialized)
    }
}
```

### Dispatcher

```rust
/// Generate a random array satisfying the given constraints.
/// May choose any compatible encoding randomly.
pub fn arbitrary_constrained_array(
    u: &mut Unstructured,
    len: Option<usize>,
    dtype: &DType,
    constraints: &ArrayConstraints,
) -> Result<ArrayRef> {
    // Collect encodings that can satisfy these constraints
    let mut candidates: Vec<ConstrainedGenerator> = vec![];

    // Always include primitive as fallback
    candidates.push(primitive_constrained_gen);

    // Add encodings that can generate directly
    if constraints.is_satisfied_by(SequenceArray::can_generate()) {
        candidates.push(sequence_constrained_gen);
    }
    if constraints.is_satisfied_by(ConstantArray::can_generate()) {
        candidates.push(constant_constrained_gen);
    }

    // Add wrapping encodings (generate base + wrap)
    if constraints.is_satisfied_by(DeltaArray::preserves()) {
        candidates.push(delta_constrained_gen);
    }
    // ... etc

    // Pick randomly and generate
    let generator = u.choose(&candidates)?;
    generator(u, len, dtype, constraints)
}
```

### Usage Example: RunEnd

```rust
impl ArbitraryRunEndArray {
    pub fn with_dtype(
        u: &mut Unstructured,
        dtype: &DType,
        len: Option<usize>,
    ) -> Result<Self> {
        let num_runs = u.int_in_range(0..=20)?;

        // Values: unconstrained, any encoding
        let values = ArbitraryArray::arbitrary_with(u, Some(num_runs), dtype)?.0;

        // Ends: strictly sorted, non-nullable, unsigned, soft target = len
        let ends = arbitrary_constrained_array(
            u,
            Some(num_runs),
            &DType::Primitive(PType::U64, Nullability::NonNullable),
            &ArrayConstraints {
                ordering: OrderingConstraint {
                    strictly_sorted: true,
                    ..default()
                },
                bounds: BoundConstraint {
                    target_max: len.map(|l| l as u64),
                    lower_bound: Some(1), // First end must be >= 1
                    ..default()
                },
                type_constraint: TypeConstraint {
                    unsigned: true,
                    ..default()
                },
                non_nullable: true,
                ..default()
            },
        )?;

        // ends could now be: Primitive, Sequence, Delta, FOR, BitPacked, etc.
        Ok(ArbitraryRunEndArray(RunEndArray::try_new(ends, values)?))
    }
}
```

### Usage Example: Sparse

```rust
impl ArbitrarySparseArray {
    pub fn with_dtype(
        u: &mut Unstructured,
        dtype: &DType,
        len: Option<usize>,
    ) -> Result<Self> {
        let len = len.unwrap_or(u.int_in_range(0..=100)?);
        let num_patches = u.int_in_range(0..=len.min(20))?;

        // Indices: strictly sorted, bounded by len
        let indices = arbitrary_constrained_array(
            u,
            Some(num_patches),
            &DType::Primitive(PType::U64, Nullability::NonNullable),
            &ArrayConstraints {
                ordering: OrderingConstraint {
                    strictly_sorted: true,
                    ..default()
                },
                bounds: BoundConstraint {
                    upper_bound: Some(len as u64),
                    ..default()
                },
                non_nullable: true,
                ..default()
            },
        )?;

        // Values: unconstrained
        let values = ArbitraryArray::arbitrary_with(u, Some(num_patches), dtype)?.0;

        // Fill value
        let fill = random_scalar(u, dtype)?;

        Ok(ArbitrarySparseArray(SparseArray::try_new(
            indices, values, len, fill
        )?))
    }
}
```

### Usage Example: Dict

```rust
impl ArbitraryDictArray {
    pub fn with_dtype(
        u: &mut Unstructured,
        dtype: &DType,
        len: Option<usize>,
    ) -> Result<Self> {
        let len = len.unwrap_or(u.int_in_range(0..=100)?);
        let dict_size = u.int_in_range(1..=20)?;

        // Dictionary values: unconstrained
        let values = ArbitraryArray::arbitrary_with(u, Some(dict_size), dtype)?.0;

        // Codes: bounded by dict_size, unsigned, non-nullable
        let codes = arbitrary_constrained_array(
            u,
            Some(len),
            &DType::Primitive(PType::U32, Nullability::NonNullable),
            &ArrayConstraints {
                bounds: BoundConstraint {
                    upper_bound: Some(dict_size as u64),
                    ..default()
                },
                type_constraint: TypeConstraint {
                    unsigned: true,
                    ..default()
                },
                non_nullable: true,
                ..default()
            },
        )?;

        Ok(ArbitraryDictArray(DictArray::try_new(codes, values)?))
    }
}
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Add `ArrayConstraints` and related types to `vortex-array/src/arrays/arbitrary.rs`
2. Implement `arbitrary_constrained_array` dispatcher
3. Implement constrained generation for `PrimitiveArray` (base case)

### Phase 2: Leaf Encodings
4. Implement `ArbitraryConstrained` for `ConstantArray`
5. Implement `ArbitraryConstrained` for `SequenceArray`

### Phase 3: Wrapping Encodings
6. Implement `ArbitraryConstrained` for `DeltaArray`
7. Implement `ArbitraryConstrained` for `FORArray`
8. Implement `ArbitraryConstrained` for `BitPackedArray`

### Phase 4: Update Complex Encodings
9. Update `ArbitraryRunEndArray` to use constrained generation
10. Update `ArbitrarySparseArray` to use constrained generation
11. Update `ArbitraryDictArray` to use constrained generation

### Phase 5: Additional Constraints
12. Add offset constraint support (for VarBin, List)
13. Add UTF-8 validity constraint support

## Alternatives Considered

### Alternative 1: Validation-based approach
Generate unconstrained arrays, then validate/fix them.

**Rejected because**: Fixing invalid data may not produce realistic distributions,
and some constraints (like sorted) are expensive to fix.

### Alternative 2: Per-encoding custom generators
Each encoding implements its own constraint handling independently.

**Rejected because**: Code duplication, inconsistent constraint handling,
harder to extend with new constraints.

### Alternative 3: Property-based generation only
Use a property testing framework's built-in constraint support.

**Rejected because**: The `arbitrary` crate doesn't have rich constraint support,
and we need composability across Vortex's encoding system.

## Open Questions

1. **Conditional preservation**: Some encodings preserve constraints only under certain
   conditions (e.g., Delta preserves sortedness only if all deltas are non-negative).
   Should we model this explicitly?

2. **Constraint composition**: When an encoding wraps another, how do we compose their
   constraints? E.g., FOR(Delta(x)) preserves what?

3. **Performance**: Should we cache/memoize the capability checks, or is the overhead negligible?

4. **Constraint conflicts**: How do we handle impossible constraint combinations
   (e.g., strictly_sorted + all values equal)?
