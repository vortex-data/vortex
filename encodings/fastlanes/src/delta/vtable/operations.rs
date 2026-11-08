// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::ProbeState;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use super::Delta;
use crate::DeltaArrayExt;
use crate::FL_CHUNK_SIZE;

/// The most recently decoded FastLanes chunk, retained across repeated reads.
///
/// Reading one value decodes the whole 1024-value chunk holding it, so a repeated read, such as a
/// binary search over RunEnd ends, would otherwise decode the same chunk once per probe.
#[derive(Default)]
pub struct DeltaProbeState {
    /// The decoded rows and their range in the array.
    chunk: Option<(Range<usize>, PrimitiveArray)>,
}

impl OperationsVTable<Delta> for Delta {
    type ProbeState = DeltaProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, Delta>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let Some(retained) = state.retained() else {
            return Self::scalar_at(array, index, ctx);
        };
        if let Some((rows, decoded)) = &retained.chunk
            && rows.contains(&index)
        {
            return decoded
                .clone()
                .into_array()
                .execute_scalar(index - rows.start, ctx);
        }
        // The rows of `index`'s FastLanes chunk, in array coordinates.
        let physical = array.offset() + index;
        let chunk_start = physical - physical % FL_CHUNK_SIZE;
        let start = chunk_start.saturating_sub(array.offset());
        let end = (chunk_start + FL_CHUNK_SIZE - array.offset()).min(array.len());
        let decoded = array
            .array()
            .slice(start..end)?
            .execute::<PrimitiveArray>(ctx)?;
        let scalar = decoded
            .clone()
            .into_array()
            .execute_scalar(index - start, ctx)?;
        retained.chunk = Some((start..end, decoded));
        Ok(scalar)
    }

    fn scalar_at(
        array: ArrayView<'_, Delta>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let decompressed = array
            .array()
            .slice(index..index + 1)?
            .execute::<PrimitiveArray>(ctx)?;
        decompressed.into_array().execute_scalar(0, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::RunEnd;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Delta;
    use crate::DeltaArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn da(array: &PrimitiveArray) -> DeltaArray {
        Delta::try_from_primitive_array(array, &mut SESSION.create_execution_ctx())
            .vortex_expect("Delta array construction should succeed")
    }

    #[test]
    fn test_slice_non_jagged_array_first_chunk_of_two() {
        let delta = da(&(0u32..2048).collect());

        let actual = delta.slice(10..250).unwrap();
        let expected = PrimitiveArray::from_iter(10u32..250).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_second_chunk_of_two() {
        let delta = da(&(0u32..2048).collect());

        let actual = delta.slice(1024 + 10..1024 + 250).unwrap();
        let expected = PrimitiveArray::from_iter((1024 + 10u32)..(1024 + 250)).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_two() {
        let delta = da(&(0u32..2048).collect());

        let actual = delta.slice(1000..1048).unwrap();
        let expected = PrimitiveArray::from_iter(1000u32..1048).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_four() {
        let delta = da(&(0u32..4096).collect());

        let actual = delta.slice(2040..2050).unwrap();
        let expected = PrimitiveArray::from_iter(2040u32..2050).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_whole() {
        let delta = da(&(0u32..4096).collect());

        let actual = delta.slice(0..4096).unwrap();
        let expected = PrimitiveArray::from_iter(0u32..4096).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_empty() {
        let delta = da(&(0u32..4096).collect());

        let actual = delta.slice(0..0).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());

        let actual = delta.slice(4096..4096).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());

        let actual = delta.slice(1024..1024).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_jagged_array_second_chunk_of_two() {
        let delta = da(&(0u32..2000).collect());

        let actual = delta.slice(1024 + 10..1024 + 250).unwrap();
        let expected = PrimitiveArray::from_iter((1024 + 10u32)..(1024 + 250)).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_jagged_array_empty() {
        let delta = da(&(0u32..4000).collect());

        let actual = delta.slice(0..0).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());

        let actual = delta.slice(4000..4000).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());

        let actual = delta.slice(1024..1024).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_of_non_jagged() {
        let delta = da(&(0u32..2048).collect());

        let sliced = delta.slice(10..1013).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![10u32, 11]).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_of_jagged() {
        let delta = da(&(0u32..2000).collect());

        let sliced = delta.slice(10..1013).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![10u32, 11]).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_non_jagged() {
        let delta = da(&(0u32..2048).collect());

        let sliced = delta.slice(1034..1050).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![1034u32, 1035]).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_jagged() {
        let delta = da(&(0u32..2000).collect());

        let sliced = delta.slice(1034..1050).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![1034u32, 1035]).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_non_jagged() {
        let delta = da(&(0u32..2048).collect());

        let sliced = delta.slice(1010..1050).unwrap();
        let sliced_again = sliced.slice(5..20).unwrap();

        let expected = PrimitiveArray::from_iter(1015u32..1030).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_jagged() {
        let delta = da(&(0u32..2000).collect());

        let sliced = delta.slice(1010..1050).unwrap();
        let sliced_again = sliced.slice(5..20).unwrap();

        let expected = PrimitiveArray::from_iter(1015u32..1030).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_scalar_at_non_jagged_array() {
        let delta = da(&(0u32..2048).collect()).into_array();

        let expected = PrimitiveArray::from_iter(0u32..2048).into_array();
        assert_arrays_eq!(delta, expected, &mut SESSION.create_execution_ctx());
        for index in [0, 1, 10, 1023, 1024, 1025, 2047] {
            let scalar = delta
                .execute_scalar(index, &mut SESSION.create_execution_ctx())
                .unwrap();
            let expected = Scalar::from(index as u32);
            assert_eq!(scalar.dtype(), expected.dtype());
            assert_eq!(scalar, expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_non_jagged_array_oob() {
        let delta = da(&(0u32..2048).collect()).into_array();
        delta
            .execute_scalar(2048, &mut SESSION.create_execution_ctx())
            .unwrap();
    }
    #[test]
    fn test_scalar_at_jagged_array() {
        let delta = da(&(0u32..2000).collect()).into_array();

        let expected = PrimitiveArray::from_iter(0u32..2000).into_array();
        assert_arrays_eq!(delta, expected, &mut SESSION.create_execution_ctx());
        for index in [0, 1, 10, 1023, 1024, 1025, 1999] {
            let scalar = delta
                .execute_scalar(index, &mut SESSION.create_execution_ctx())
                .unwrap();
            let expected = Scalar::from(index as u32);
            assert_eq!(scalar.dtype(), expected.dtype());
            assert_eq!(scalar, expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_jagged_array_oob() {
        let delta = da(&(0u32..2000).collect()).into_array();
        delta
            .execute_scalar(2000, &mut SESSION.create_execution_ctx())
            .unwrap();
    }

    #[rstest]
    // Basic delta arrays
    #[case::delta_u32((0u32..100).collect())]
    #[case::delta_u64((0..100).map(|i| i as u64 * 10).collect())]
    // Large arrays (multiple chunks)
    #[case::delta_large_u32((0u32..2048).collect())]
    #[case::delta_large_u64((0u64..2048).collect())]
    // Single element
    #[case::delta_single(PrimitiveArray::new(buffer![42u32], Validity::NonNullable))]
    // Signed inputs (added with signed-delta support).
    #[case::delta_i32_crossing_zero((-100i32..100).collect())]
    #[case::delta_i64_negative((0i64..100).map(|i| -i * 10).collect())]
    #[case::delta_large_i32((-1024i32..1024).collect())]
    #[case::delta_single_negative(PrimitiveArray::new(buffer![-42i32], Validity::NonNullable))]
    fn test_delta_consistency(#[case] array: PrimitiveArray) {
        test_array_consistency(
            &da(&array).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
    }

    #[rstest]
    #[case::delta_u8_basic(PrimitiveArray::new(buffer![1u8, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u16_basic(PrimitiveArray::new(buffer![1u16, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u32_basic(PrimitiveArray::new(buffer![1u32, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u64_basic(PrimitiveArray::new(buffer![1u64, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u32_large(PrimitiveArray::new(buffer![1u32; 100], Validity::NonNullable))]
    #[case::delta_i8_basic(PrimitiveArray::new(buffer![-1i8, -1, -1, -1, -1], Validity::NonNullable))]
    #[case::delta_i32_basic(PrimitiveArray::new(buffer![-1i32, -1, -1, -1, -1], Validity::NonNullable))]
    fn test_delta_binary_numeric(#[case] array: PrimitiveArray) {
        test_binary_numeric_array(
            &da(&array).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
    }

    /// Repeated reads keep the last decoded chunk. Reads in any order, across chunk boundaries,
    /// through nulls and from an offset slice must still match one-off reads.
    #[rstest]
    #[case::whole(0..5000)]
    #[case::offset_slice(1000..4500)]
    #[case::within_one_chunk(2050..2060)]
    fn repeated_probe_matches_one_off_reads(#[case] range: Range<usize>) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_option_iter(
            (0..5000u32).map(|i| (i % 7 != 3).then_some(i * 3 + i % 5)),
        );
        let delta = da(&values).into_array().slice(range)?;
        let len = delta.len();
        let mut probe = delta.repeated_probe();
        // Jump between chunks and back, as a binary search does.
        let order = (0..len)
            .step_by(997)
            .chain((0..len).rev().step_by(13))
            .chain(0..len);
        for i in order {
            assert_eq!(
                probe.execute_scalar(i, &mut ctx)?,
                delta.execute_scalar(i, &mut ctx)?
            );
        }
        Ok(())
    }

    /// Slicing a RunEnd array searches its ends through a repeated probe; with Delta-encoded
    /// ends, that search reads Delta chunks through the retained state.
    #[rstest]
    #[case::first_chunk(0..900)]
    #[case::middle(3000..9000)]
    #[case::unaligned_tail(7777..12000)]
    fn runend_slice_over_delta_ends(#[case] range: Range<usize>) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 3000 runs of 4 rows, so the ends span three Delta chunks.
        let ends: PrimitiveArray = (1..=3000u32).map(|i| i * 4).collect();
        let values: PrimitiveArray = (0..3000i32).map(|i| i % 17 - 8).collect();
        let runend = RunEnd::try_new(da(&ends).into_array(), values.into_array(), &mut ctx)?;
        let expected: PrimitiveArray = (0..12000usize)
            .map(|row| i32::try_from(row / 4 % 17).vortex_expect("small") - 8)
            .collect();
        assert_arrays_eq!(
            runend.into_array().slice(range.clone())?,
            expected.into_array().slice(range)?,
            &mut ctx
        );
        Ok(())
    }
}
