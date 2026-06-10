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
fn gather<W: NativePType, A: AsPrimitive<usize>, R: AsPrimitive<usize>>(
    values: &[&[W]],
    branches: &[A],
    rows: &[R],
) -> VortexResult<BufferMut<W>> {
    let len = branches.len();

    // Validate the per-row bounds once up front (returning an error rather than panicking), so the
    // bounds checks in the gather passes below always pass and predict perfectly.
    for i in 0..len {
        let branch = branches[i].as_();
        vortex_ensure!(
            branch < values.len(),
            "interleave array index out of bounds"
        );
        vortex_ensure!(
            rows[i].as_() < values[branch].len(),
            "interleave row index out of bounds"
        );
    }

    // Process 8 elements at a time, issuing 8 independent loads before any store: a random-access
    // gather is latency-bound, and this keeps multiple cache misses in flight (memory-level
    // parallelism) instead of serializing on each one.
    let mut out = BufferMut::<W>::zeroed(len);
    let mut out_chunks = out.chunks_exact_mut(8);
    let idx_chunks = branches.chunks_exact(8).zip(rows.chunks_exact(8));
    for (out_chunk, (b, r)) in (&mut out_chunks).zip(idx_chunks) {
        let v0 = values[b[0].as_()][r[0].as_()];
        let v1 = values[b[1].as_()][r[1].as_()];
        let v2 = values[b[2].as_()][r[2].as_()];
        let v3 = values[b[3].as_()][r[3].as_()];
        let v4 = values[b[4].as_()][r[4].as_()];
        let v5 = values[b[5].as_()][r[5].as_()];
        let v6 = values[b[6].as_()][r[6].as_()];
        let v7 = values[b[7].as_()][r[7].as_()];
        out_chunk[0] = v0;
        out_chunk[1] = v1;
        out_chunk[2] = v2;
        out_chunk[3] = v3;
        out_chunk[4] = v4;
        out_chunk[5] = v5;
        out_chunk[6] = v6;
        out_chunk[7] = v7;
    }

    let tail = out_chunks.into_remainder();
    let start = len - tail.len();
    for (slot, i) in tail.iter_mut().zip(start..len) {
        *slot = values[branches[i].as_()][rows[i].as_()];
    }

    Ok(out)
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
