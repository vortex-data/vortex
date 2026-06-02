// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::FL_ORDER;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::Delta;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::lane_count;

impl OperationsVTable<Delta> for Delta {
    /// Reconstruct a single value without decompressing the whole chunk.
    ///
    /// Delta decoding is independent per FastLanes lane: a value at logical position `p` lives in
    /// exactly one lane and is the prefix sum of that lane's deltas, seeded by the lane base, up to
    /// its row. We therefore materialize only the 1,024-element chunk containing `index` and walk
    /// the single relevant lane, rather than running the full undelta + untranspose over all 1,024
    /// values as canonicalization would.
    fn scalar_at(
        array: ArrayView<'_, Delta>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        vortex_ensure!(
            index < array.len(),
            "index {index} out of bounds for Delta array of length {}",
            array.len()
        );

        let ptype = array.dtype().as_ptype();
        let lanes = lane_count(ptype);
        let rows = 1024 / lanes;

        // Resolve the physical position within the chunk that backs this value.
        let physical = index + array.offset();
        let chunk = physical / 1024;
        let chunk_local = physical % 1024;

        // Materialize only this chunk's bases and deltas. Validity is handled by the generic
        // `execute_scalar` guard before dispatch, so (as in bitpacking) we only reconstruct the
        // value here.
        let bases = array
            .bases()
            .slice(chunk * lanes..(chunk + 1) * lanes)?
            .execute::<PrimitiveArray>(ctx)?;
        let deltas = array
            .deltas()
            .slice(chunk * 1024..(chunk + 1) * 1024)?
            .execute::<PrimitiveArray>(ctx)?;

        // Position of this value within the FastLanes-transposed (delta-encoded) value buffer.
        let transposed = untranspose_index(chunk_local);

        // Accumulate the lane's deltas onto its base up to (and including) this value's row.
        // `wrapping_add` recovers both signed and unsigned values, inverting the `wrapping_sub`
        // performed at compress time.
        let lane = (transposed % 128) % lanes;
        let scalar = match_each_integer_ptype!(ptype, |P| {
            let bases = bases.as_slice::<P>();
            let deltas = deltas.as_slice::<P>();
            let mut value = bases[lane];
            for row in 0..rows {
                let idx = transposed_lane_index(row, lane);
                value = value.wrapping_add(deltas[idx]);
                if idx == transposed {
                    break;
                }
            }
            Scalar::primitive(value, array.dtype().nullability())
        });

        Ok(scalar)
    }
}

/// Position of the `row`th value of `lane` within the FastLanes-transposed 1,024-element buffer.
///
/// This matches the per-lane iteration order used by `fastlanes` delta (un)packing.
fn transposed_lane_index(row: usize, lane: usize) -> usize {
    FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane
}

/// Map a logical chunk-local index to its slot in the FastLanes-transposed buffer.
///
/// This is the inverse of `fastlanes::transpose`.
fn untranspose_index(idx: usize) -> usize {
    let lane = idx / 64;
    let rem = idx % 64;
    let order = FL_ORDER[rem / 8];
    let row = rem % 8;
    row * 128 + order * 16 + lane
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use crate::Delta;
    use crate::DeltaArray;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn da(array: &PrimitiveArray) -> DeltaArray {
        Delta::try_from_primitive_array(array, &mut SESSION.create_execution_ctx())
            .vortex_expect("Delta array construction should succeed")
    }

    #[test]
    fn test_slice_non_jagged_array_first_chunk_of_two() {
        let delta = da(&(0u32..2048).collect());

        let actual = delta.slice(10..250).unwrap();
        let expected = PrimitiveArray::from_iter(10u32..250).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_second_chunk_of_two() {
        let delta = da(&(0u32..2048).collect());

        let actual = delta.slice(1024 + 10..1024 + 250).unwrap();
        let expected = PrimitiveArray::from_iter((1024 + 10u32)..(1024 + 250)).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_two() {
        let delta = da(&(0u32..2048).collect());

        let actual = delta.slice(1000..1048).unwrap();
        let expected = PrimitiveArray::from_iter(1000u32..1048).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_four() {
        let delta = da(&(0u32..4096).collect());

        let actual = delta.slice(2040..2050).unwrap();
        let expected = PrimitiveArray::from_iter(2040u32..2050).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_whole() {
        let delta = da(&(0u32..4096).collect());

        let actual = delta.slice(0..4096).unwrap();
        let expected = PrimitiveArray::from_iter(0u32..4096).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_non_jagged_array_empty() {
        let delta = da(&(0u32..4096).collect());

        let actual = delta.slice(0..0).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected);

        let actual = delta.slice(4096..4096).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected);

        let actual = delta.slice(1024..1024).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_jagged_array_second_chunk_of_two() {
        let delta = da(&(0u32..2000).collect());

        let actual = delta.slice(1024 + 10..1024 + 250).unwrap();
        let expected = PrimitiveArray::from_iter((1024 + 10u32)..(1024 + 250)).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_jagged_array_empty() {
        let delta = da(&(0u32..4000).collect());

        let actual = delta.slice(0..0).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected);

        let actual = delta.slice(4000..4000).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected);

        let actual = delta.slice(1024..1024).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected);
    }

    #[test]
    fn test_slice_of_slice_of_non_jagged() {
        let delta = da(&(0u32..2048).collect());

        let sliced = delta.slice(10..1013).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![10u32, 11]).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_of_jagged() {
        let delta = da(&(0u32..2000).collect());

        let sliced = delta.slice(10..1013).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![10u32, 11]).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_non_jagged() {
        let delta = da(&(0u32..2048).collect());

        let sliced = delta.slice(1034..1050).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![1034u32, 1035]).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_jagged() {
        let delta = da(&(0u32..2000).collect());

        let sliced = delta.slice(1034..1050).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![1034u32, 1035]).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_non_jagged() {
        let delta = da(&(0u32..2048).collect());

        let sliced = delta.slice(1010..1050).unwrap();
        let sliced_again = sliced.slice(5..20).unwrap();

        let expected = PrimitiveArray::from_iter(1015u32..1030).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_jagged() {
        let delta = da(&(0u32..2000).collect());

        let sliced = delta.slice(1010..1050).unwrap();
        let sliced_again = sliced.slice(5..20).unwrap();

        let expected = PrimitiveArray::from_iter(1015u32..1030).into_array();
        assert_arrays_eq!(sliced_again, expected);
    }

    #[test]
    fn test_scalar_at_non_jagged_array() {
        let delta = da(&(0u32..2048).collect()).into_array();

        let expected = PrimitiveArray::from_iter(0u32..2048).into_array();
        assert_arrays_eq!(delta, expected);
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
        assert_arrays_eq!(delta, expected);
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
        test_array_consistency(&da(&array).into_array());
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
        test_binary_numeric_array(da(&array).into_array());
    }

    /// `untranspose_index` must invert `fastlanes::transpose` over a full 1,024-element vector.
    #[test]
    fn untranspose_index_inverts_transpose() {
        for i in 0..1024 {
            assert_eq!(super::untranspose_index(fastlanes::transpose(i)), i);
            assert_eq!(fastlanes::transpose(super::untranspose_index(i)), i);
        }
    }

    /// `scalar_at` at every index must agree with the fully decompressed (canonical) array.
    fn check_scalar_at(array: PrimitiveArray) {
        let expected = array.clone().into_array();
        let delta = da(&array).into_array();
        assert_eq!(delta.len(), expected.len());

        for i in 0..delta.len() {
            let got = delta
                .execute_scalar(i, &mut SESSION.create_execution_ctx())
                .vortex_expect("delta scalar_at");
            let want = expected
                .execute_scalar(i, &mut SESSION.create_execution_ctx())
                .vortex_expect("reference scalar_at");
            assert_eq!(got.is_valid(), want.is_valid(), "validity mismatch at {i}");
            if want.is_valid() {
                assert_eq!(got, want, "value mismatch at {i}");
            }
        }
    }

    #[rstest]
    #[case::u8((0u8..200).collect())]
    #[case::u16((0u16..1500).map(|i| i * 3).collect())]
    #[case::u32((0u32..2050).collect())]
    #[case::u64((0u64..2050).map(|i| i * 7).collect())]
    #[case::i32_crossing_zero((-1100i32..1100).collect())]
    #[case::i64_negative((0i64..2050).map(|i| -i * 5).collect())]
    #[case::single(PrimitiveArray::new(buffer![42u32], Validity::NonNullable))]
    #[case::chunk_boundary((0u32..1025).collect())]
    fn test_scalar_at_matches_canonical(#[case] array: PrimitiveArray) {
        check_scalar_at(array);
    }

    #[rstest]
    #[case::nullable_u32(PrimitiveArray::from_option_iter(
        (0u32..1100).map(|i| (i % 3 != 0).then_some(i)),
    ))]
    #[case::nullable_i64(PrimitiveArray::from_option_iter(
        (0i64..1100).map(|i| (i % 5 != 0).then_some(-i * 2)),
    ))]
    fn test_scalar_at_matches_canonical_nullable(#[case] array: PrimitiveArray) {
        check_scalar_at(array);
    }

    /// `scalar_at` on a sliced array must honor the physical offset across chunk boundaries.
    #[test]
    fn test_scalar_at_sliced_offset() {
        let delta = da(&(0u32..2048).collect()).into_array();
        let sliced = delta.slice(1000..1100).unwrap();
        let expected = PrimitiveArray::from_iter(1000u32..1100).into_array();

        for i in 0..sliced.len() {
            let got = sliced
                .execute_scalar(i, &mut SESSION.create_execution_ctx())
                .vortex_expect("delta scalar_at");
            let want = expected
                .execute_scalar(i, &mut SESSION.create_execution_ctx())
                .vortex_expect("reference scalar_at");
            assert_eq!(got, want, "value mismatch at {i}");
        }
    }
}
