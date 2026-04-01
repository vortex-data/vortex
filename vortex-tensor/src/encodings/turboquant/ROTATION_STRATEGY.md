# Non-Power-of-2 Rotation Strategy for TurboQuant

## Problem Statement

The SRHT requires zero-padding to the next power of 2. For non-power-of-2 dims, the
zero-padded entries cause a distribution mismatch that elevates QJL bias from ~11% to
~23%+ and worsens with smaller dimensions. The fix is to use a rotation that produces
the correct coordinate distribution without zero-padding.

## Approach: Tiered rotation by dimension structure

Three tiers based on what the dimension actually is:

| Dimension structure | Example dims | Rotation | Rationale |
|---------------------|-------------|----------|-----------|
| Power of 2 | 128, 256, 512, 1024 | SRHT (current) | No padding, exact distribution |
| Sum of 2 powers of 2 (>128) | 384, 768, 1536 | Split SRHT | Two independent SRHTs, no padding |
| Small (≤128) non-power-of-2 | 96, 100, 112 | Dense orthogonal | d² is cheap at small d |
| Other (>128) | 837, 1000 | SRHT with padding | Accept QJL bias, current behavior |

The key insight: the common non-power-of-2 embedding dimensions (768, 384, 1536) are
almost always sums of two powers of two. We can exploit this structure directly.

## Split SRHT for sum-of-two-powers dimensions

For dim = 2^a + 2^b (e.g., 768 = 512 + 256):

1. Split the d-dimensional vector into two chunks: `x[0..2^a]` and `x[2^a..d]`
2. Apply independent SRHTs of size 2^a and 2^b to each chunk
3. Concatenate the results → d rotated coordinates (no padding!)

**Properties:**
- Each chunk is power-of-2 → SRHT produces the exact analytical distribution
- Centroids use `d` with the standard formula → MSE within theoretical bound
- QJL scale uses `d` → correct inner product estimation
- Compute: O(2^a × log(2^a) + 2^b × log(2^b)) ≈ O(d log d) — same as SRHT
- Storage: 3×2^a + 3×2^b = 3d sign bits — same as SRHT

**Missing cross-chunk mixing:** The two SRHTs don't mix information between the halves.
If the original vector has energy concentrated in one half, the rotation quality degrades.
Fix: apply a random coordinate permutation before splitting, spreading the energy.
The permutation is O(d) and needs d×ceil(log2(d)) bits of storage (~1.3 KB for d=768).

**Full pipeline:**
1. Permute the d-dimensional vector (scatter energy across both halves)
2. Split into two power-of-2 chunks
3. Apply independent SRHTs to each chunk
4. Concatenate → d rotated coordinates
5. Quantize with d-dimensional centroids

## Dense orthogonal rotation for small dimensions (≤128)

For d ≤ 128, generate a random d×d orthogonal matrix Q via QR of Gaussian.
- d=128: Q is 128² × 4 = 64 KB (acceptable)
- d=96: Q is 96² × 4 = 36 KB
- Rotate via dense GEMV: 128² = 16K FLOPS (vs SRHT's ~2.7K — 6× more, but small absolute cost)

## Implementation Plan

### Step 1: Identify rotation strategy at encode time

Add a function that classifies the dimension:

```rust
enum RotationKind {
    /// dim is a power of 2. Use standard SRHT.
    Srht,
    /// dim = 2^a + 2^b with a > b. Use permutation + split SRHTs.
    SplitSrht { high: usize, low: usize },
    /// dim ≤ 128 and non-power-of-2. Use dense d×d orthogonal matrix.
    Dense,
    /// dim > 128, not a power of 2, not sum of two powers. Use SRHT with padding.
    SrhtPadded,
}

fn classify_dimension(dim: usize) -> RotationKind {
    if dim.is_power_of_two() {
        return RotationKind::Srht;
    }
    if dim <= 128 {
        return RotationKind::Dense;
    }
    // Check if dim = 2^a + 2^b for some a > b.
    // Equivalently: dim has exactly two set bits in binary representation.
    if dim.count_ones() == 2 {
        let low = 1 << dim.trailing_zeros();
        let high = dim - low;
        return RotationKind::SplitSrht { high, low };
    }
    RotationKind::SrhtPadded
}
```

### Step 2: Implement `SplitSrhtRotation` in rotation.rs

```rust
pub struct SplitSrhtRotation {
    permutation: Vec<u16>,
    inverse_permutation: Vec<u16>,
    high_srht: SrhtRotation,  // operates on first 2^a elements
    low_srht: SrhtRotation,   // operates on last 2^b elements
    split_point: usize,       // = 2^a (= high)
    dimension: usize,         // = 2^a + 2^b
}
```

**`rotate(input, output)`:**
1. Apply permutation: `scratch[perm[i]] = input[i]`
2. Apply `high_srht.rotate(scratch[0..split], output[0..split])`
3. Apply `low_srht.rotate(scratch[split..dim], output[split..dim])`

**`inverse_rotate(input, output)`:**
1. Apply `high_srht.inverse_rotate(input[0..split], scratch[0..split])`
2. Apply `low_srht.inverse_rotate(input[split..dim], scratch[split..dim])`
3. Apply inverse permutation: `output[inv_perm[i]] = scratch[i]`

**Storage:** 3×high + 3×low sign bits (= 3×dim total) + dim permutation indices.
Stored as children: two rotation_signs arrays + one permutation array.

### Step 3: Implement `DenseRotation` in rotation.rs

```rust
pub struct DenseRotation {
    matrix: Vec<f32>,   // d×d row-major orthogonal matrix
    dimension: usize,
}
```

- `try_new(seed, dim)`: Generate Gaussian d×d, QR factorize, keep Q
- `rotate`: dense GEMV
- `inverse_rotate`: dense GEMV with transposed Q
- Storage: d² × f32 as a child array

### Step 4: Unify under `Rotation` enum

```rust
pub enum Rotation {
    Srht(SrhtRotation),
    SplitSrht(SplitSrhtRotation),
    Dense(DenseRotation),
    SrhtPadded(SrhtRotation),  // current behavior for arbitrary dims
}
```

All variants implement `rotate(input, output)` and `inverse_rotate(input, output)`.
The `Srht` and `SrhtPadded` variants use padded buffers; `SplitSrht` and `Dense`
operate in d dimensions directly.

### Step 5: Update metadata and slots

Add `rotation_type: u32` to `TurboQuantMetadata` (tag 5, default 0 = SRHT/SrhtPadded
for backward compat). Values: 0=SRHT, 1=SplitSrht, 2=Dense.

Slot layout depends on rotation type:
- SRHT: slot 3 = rotation_signs (3×padded_dim, unchanged)
- SplitSrht: slot 3 = high_signs (3×high), new slots for low_signs + permutation
- Dense: slot 3 = matrix (d² × f32)

### Step 6: Update compress/decompress

For SplitSrht and Dense rotations:
- Centroids use `d` (not padded_dim) → standard analytical formula
- QJL scale uses `d` → correct inner product estimation
- No zero-padding buffers needed (operate in d dimensions)
- No pad-position residual handling needed

### Step 7: Tests

- Power-of-2: unchanged (SRHT path)
- 768, 384, 1536: SplitSrht path, 0.15 QJL bias, MSE within theoretical bound
- Small non-power-of-2 (96): Dense path, same quality guarantees
- Arbitrary dims (837): SrhtPadded, 0.25 QJL bias threshold (current behavior)
- Backward compat: `rotation_type=0` decodes identically to current

## Key Design Decisions

**Why permute before split?** Without permutation, if the embedding model puts
different features in different halves of the vector, one SRHT might get much more
variance than the other. The permutation ensures both halves get a uniform mix of
the original dimensions, so both SRHTs see statistically similar inputs.

**Why not split for arbitrary dims?** A dimension like 837 doesn't decompose into
two powers of two. We could decompose into more terms (837 = 512 + 256 + 64 + 4 + 1)
but many small SRHTs lose mixing quality. The SRHT-with-padding approach is acceptable
for these rare cases.

**Why dense only for ≤128?** At d=128, the dense matrix is 64 KB and GEMV is 16K
FLOPS — both small. At d=768, it's 2.36 MB and 590K FLOPS — the storage is
significant and the compute gap widens. The split SRHT gives O(d log d) for
the common large non-power-of-2 dims.

## What we tried and learned

| Approach | 768/3-bit QJL bias | 768/4-bit QJL bias | 768/8-bit MSE | Verdict |
|----------|-------------------|-------------------|---------------|---------|
| Original (padded_dim centroids) | -0.24 | -0.22 | within bound | baseline |
| Analytical (dim centroids) | -0.15 | -0.28 | within bound | mixed |
| MC empirical centroids | passes 0.15 | +0.06 | 25× over bound | MSE regression |
| Random permutation before SRHT | -0.24 | -0.22 | within bound | no effect |

Key takeaways:
- The bias is caused by distribution mismatch from zero-padding, not centroid tuning
- MC centroids optimize for the actual distribution but violate the theoretical MSE bound
- Fixing centroids alone trades MSE quality for QJL bias — a fundamental tension
- The principled fix is to eliminate the distribution mismatch at the rotation level

## Verification

1. All existing tests pass (SRHT path unchanged for power-of-2)
2. 768/384/1536 pass at 0.15 QJL bias (SplitSrht path)
3. MSE within theoretical bound for all rotation types
4. Benchmarks: SplitSrht throughput comparable to SRHT
5. Backward compat: old files with rotation_type=0 decode correctly
