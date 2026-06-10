// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Primitive-value execution: the optimized [`Interleave`](super::super::Interleave) path for
//! primitive values.

use num_traits::AsPrimitive;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use crate::array::Array;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_unsigned_integer_ptype;
use crate::require_child;
use crate::validity::Validity;

/// Gathers `N` primitive values under unsigned `array_indices` / `row_indices` selectors.
///
/// The gather moves fixed-width elements and never inspects them, so every value ptype of a given
/// byte width shares one kernel: the values are zero-copy reinterpreted to the same-width unsigned
/// type, gathered, and the output reinterpreted back. For floats this round-trips raw bit
/// patterns — pure bit movement with no arithmetic — so it is semantics-preserving (including for
/// NaNs).
pub(super) fn execute(
    array: Array<Interleave>,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    // Validity is gathered separately from the values and is not yet implemented for primitives;
    // the output is nullable iff any value is nullable, so this also guarantees that every value
    // (and hence its width-reinterpretation below) is non-nullable.
    vortex_ensure!(
        !array.as_ref().dtype().is_nullable(),
        "interleave execution for nullable primitive values is not yet implemented"
    );

    let num_values = array.num_values();

    // Drive every value and both selectors to canonical encodings so we can operate on raw slices.
    let mut array = array;
    for i in 0..num_values {
        array = require_child!(array, array.value(i), i => Primitive);
    }
    array = require_child!(array, array.array_indices(), num_values => Primitive);
    array = require_child!(array, array.row_indices(), num_values + 1 => Primitive);

    let ptype = match array.as_ref().dtype() {
        DType::Primitive(ptype, _) => *ptype,
        dtype => vortex_panic!("interleave primitive kernel on non-primitive dtype {dtype}"),
    };
    let gather_ptype = match ptype {
        PType::F16 => PType::U16,
        PType::F32 => PType::U32,
        PType::F64 => PType::U64,
        p => p.to_unsigned(),
    };

    let values: Vec<PrimitiveArray> = (0..num_values)
        .map(|i| {
            array
                .value(i)
                .as_::<Primitive>()
                .reinterpret_cast(gather_ptype)
        })
        .collect();

    // Gather directly from the typed selector buffers — no intermediate `usize` materialization.
    let array_indices = array.array_indices().as_::<Primitive>();
    let row_indices = array.row_indices().as_::<Primitive>();
    let gathered = match_each_unsigned_integer_ptype!(gather_ptype, |W| {
        let value_slices: Vec<&[W]> = values.iter().map(|v| v.as_slice::<W>()).collect();
        match_each_unsigned_integer_ptype!(array_indices.ptype(), |A| {
            match_each_unsigned_integer_ptype!(row_indices.ptype(), |R| {
                PrimitiveArray::new(
                    gather(
                        &value_slices,
                        array_indices.as_slice::<A>(),
                        row_indices.as_slice::<R>(),
                    )?
                    .freeze(),
                    Validity::NonNullable,
                )
            })
        })
    });

    Ok(ExecutionResult::done(gathered.reinterpret_cast(ptype)))
}

/// The gather, monomorphized on the value width and the selector integer widths so each element
/// and `(array_index, row_index)` pair is read straight from its packed buffer.
///
/// The loop body is deliberately a plain zipped `collect`: `BufferMut`'s size-hinted extend
/// writes through a raw pointer with no per-item capacity check, and out-of-order execution
/// already overlaps the random-access loads across iterations. A manually unrolled
/// "N independent loads then N stores" variant (as in arrow-rs) measured *slower* here because
/// the in-loop bounds checks are potential panics whose order the compiler must preserve,
/// turning the unroll into eight in-flight check chains and register spills.
fn gather<W: NativePType, A: AsPrimitive<usize> + Ord, R: AsPrimitive<usize> + Ord>(
    values: &[&[W]],
    branches: &[A],
    rows: &[R],
) -> VortexResult<BufferMut<W>> {
    validate_bounds(values, branches, rows)?;
    Ok(branches
        .iter()
        .zip(rows)
        .map(|(b, r)| values[b.as_()][r.as_()])
        .collect())
}

/// Validates every `(array_index, row_index)` pair, returning an error rather than panicking.
///
/// The hot path is a branchless max-fold over both selectors, which auto-vectorizes (an
/// early-exit per-row check measured ~40% of the whole gather's runtime). `max_row < min(len)`
/// proves all rows in bounds exactly when the values share a length; only ragged value lengths
/// fall back to the exact per-row scan.
fn validate_bounds<W: NativePType, A: AsPrimitive<usize> + Ord, R: AsPrimitive<usize> + Ord>(
    values: &[&[W]],
    branches: &[A],
    rows: &[R],
) -> VortexResult<()> {
    if branches.is_empty() {
        return Ok(());
    }
    // Fold in the native selector type so the lanes stay narrow enough to vectorize.
    let mut max_branch = branches[0];
    let mut max_row = rows[0];
    for (b, r) in branches.iter().zip(rows) {
        max_branch = max_branch.max(*b);
        max_row = max_row.max(*r);
    }
    vortex_ensure!(
        max_branch.as_() < values.len(),
        "interleave array index out of bounds"
    );

    let min_len = values.iter().map(|v| v.len()).min().unwrap_or(0);
    if max_row.as_() < min_len {
        return Ok(());
    }
    for (b, r) in branches.iter().zip(rows) {
        vortex_ensure!(
            r.as_() < values[b.as_()].len(),
            "interleave row index out of bounds"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::super::super::InterleaveArray;
    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::NativePType;
    use crate::dtype::half::f16;

    /// Builds an `InterleaveArray` over `branches` from per-output `(array_index, row_index)`
    /// pairs. The pairs may be out of bounds: that is a runtime precondition checked at execution.
    fn build<T: NativePType>(
        branches: &[Vec<T>],
        indices: &[(usize, usize)],
    ) -> VortexResult<ArrayRef> {
        let values: Vec<ArrayRef> = branches
            .iter()
            .map(|b| PrimitiveArray::from_iter(b.iter().copied()).into_array())
            .collect();
        let array_indices = PrimitiveArray::from_iter(
            indices
                .iter()
                .map(|&(a, _)| u32::try_from(a).vortex_expect("array index fits in u32")),
        )
        .into_array();
        let row_indices = PrimitiveArray::from_iter(
            indices
                .iter()
                .map(|&(_, r)| u32::try_from(r).vortex_expect("row index fits in u32")),
        )
        .into_array();
        Ok(InterleaveArray::try_new(values, array_indices, row_indices)?.into_array())
    }

    /// Asserts that the optimized execute path matches a gather computed directly from the source
    /// vectors, exercising construction, `execute`, and `scalar_at` (via `assert_arrays_eq`).
    fn check<T: NativePType>(branches: &[Vec<T>], indices: &[(usize, usize)]) -> VortexResult<()> {
        let interleaved = build(branches, indices)?;
        let expected =
            PrimitiveArray::from_iter(indices.iter().map(|&(a, r)| branches[a][r])).into_array();
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }

    #[test]
    fn interleave_reorders_and_repeats() -> VortexResult<()> {
        // Random access: rows are pulled out of order and branch 0 row 0 is repeated.
        let indices = [(0, 1), (1, 0), (0, 0), (1, 1), (0, 0)];
        check::<u8>(&[vec![1, 2], vec![10, 20]], &indices)?;
        check::<i32>(&[vec![-1, 2], vec![10, -20]], &indices)?;
        check::<f64>(&[vec![0.5, -2.0], vec![f64::NAN, 20.25]], &indices)?;
        check::<f16>(
            &[
                vec![f16::from_f32(0.5), f16::from_f32(-2.0)],
                vec![f16::from_f32(8.0), f16::from_f32(20.25)],
            ],
            &indices,
        )
    }

    #[test]
    fn interleave_longer_than_unroll() -> VortexResult<()> {
        // 19 rows: exercises both the 8-wide unrolled chunks and the 3-row remainder.
        let branches = vec![
            (0..7i64).collect::<Vec<_>>(),
            (100..105i64).collect::<Vec<_>>(),
            (1000..1003i64).collect::<Vec<_>>(),
        ];
        let indices: Vec<(usize, usize)> = (0..19)
            .map(|i| {
                let a = i % 3;
                (a, (i * 5 + 1) % branches[a].len())
            })
            .collect();
        check(&branches, &indices)
    }

    #[test]
    fn interleave_empty() -> VortexResult<()> {
        check::<u16>(&[vec![1], vec![2]], &[])
    }

    #[test]
    fn interleave_non_canonical_children() -> VortexResult<()> {
        // A constant value array and mixed-width selectors: the kernel must canonicalize all
        // children before gathering.
        let v0 = ConstantArray::new(7i32, 3).into_array();
        let v1 = PrimitiveArray::from_iter([100i32, 200]).into_array();
        let array_indices = PrimitiveArray::from_iter([0u8, 1, 0, 1]).into_array();
        let row_indices = PrimitiveArray::from_iter([2u64, 0, 0, 1]).into_array();
        let interleaved =
            InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?.into_array();
        let expected = PrimitiveArray::from_iter([7i32, 100, 7, 200]).into_array();
        assert_arrays_eq!(interleaved, expected);
        Ok(())
    }

    #[test]
    fn rejects_out_of_bounds_array_index() -> VortexResult<()> {
        let interleaved = build::<i32>(&[vec![1, 2], vec![3, 4]], &[(0, 0), (5, 0)])?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let err = interleaved
            .execute::<Canonical>(&mut ctx)
            .err()
            .vortex_expect("expected an out-of-bounds array index to error");
        assert!(
            err.to_string().contains("array index out of bounds"),
            "{err}"
        );
        Ok(())
    }

    #[test]
    fn rejects_out_of_bounds_row_index() -> VortexResult<()> {
        let interleaved = build::<i32>(&[vec![1, 2], vec![3, 4]], &[(0, 0), (1, 9)])?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let err = interleaved
            .execute::<Canonical>(&mut ctx)
            .err()
            .vortex_expect("expected an out-of-bounds row index to error");
        assert!(err.to_string().contains("row index out of bounds"), "{err}");
        Ok(())
    }

    #[test]
    fn rejects_nullable_values_for_now() -> VortexResult<()> {
        let v0 = PrimitiveArray::from_option_iter([Some(1i32), None]).into_array();
        let v1 = PrimitiveArray::from_option_iter([Some(3i32)]).into_array();
        let array_indices = PrimitiveArray::from_iter([0u32, 1]).into_array();
        let row_indices = PrimitiveArray::from_iter([0u32, 0]).into_array();
        let interleaved =
            InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)?.into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let err = interleaved
            .execute::<Canonical>(&mut ctx)
            .err()
            .vortex_expect("expected nullable primitive values to be unimplemented");
        assert!(err.to_string().contains("not yet implemented"), "{err}");
        Ok(())
    }
}
