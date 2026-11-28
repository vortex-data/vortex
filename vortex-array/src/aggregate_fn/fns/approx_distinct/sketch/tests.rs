// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use rstest::rstest;
use vortex_error::VortexResult;

use super::HyperLogLog;
use super::MAX_PRECISION;
use super::MIN_PRECISION;
use crate::test_harness::check_metadata;

const PRECISION: u8 = 12;

fn sketch_of(precision: u8, values: Range<i64>) -> VortexResult<HyperLogLog> {
    let mut sketch = HyperLogLog::try_new(precision)?;
    for value in values {
        sketch.insert(&value.to_le_bytes());
    }
    Ok(sketch)
}

/// A hash for register `index` with rank `rank`; a rank past the remaining bits leaves them zero.
fn hash_for(precision: u8, index: u64, rank: u32) -> u64 {
    let remaining_bits = 64 - u32::from(precision);
    let index_bits = index << remaining_bits;
    if rank > remaining_bits {
        index_bits
    } else {
        index_bits | (1u64 << (remaining_bits - rank))
    }
}

fn assert_within_three_standard_errors(estimate: u64, exact: u64, precision: u8) {
    let standard_error = 1.04 / f64::from(1u32 << precision).sqrt();
    let allowed = (3.0 * standard_error * exact as f64).max(1.0);
    let difference = (estimate as f64 - exact as f64).abs();
    assert!(
        difference <= allowed,
        "estimate {estimate} for {exact} values at precision {precision} is off by {difference}, \
         more than {allowed}"
    );
}

#[rstest]
#[case::zero(0)]
#[case::below_minimum(MIN_PRECISION - 1)]
#[case::above_maximum(MAX_PRECISION + 1)]
#[case::eighteen(18)]
#[case::max_byte(u8::MAX)]
fn test_try_new_rejects_out_of_range_precision(#[case] precision: u8) {
    assert!(HyperLogLog::try_new(precision).is_err());
}

#[rstest]
fn test_try_new_allocates_one_register_per_index(
    #[values(MIN_PRECISION, PRECISION, MAX_PRECISION)] precision: u8,
) -> VortexResult<()> {
    let sketch = HyperLogLog::try_new(precision)?;
    assert_eq!(sketch.precision(), precision);
    assert_eq!(sketch.registers.len(), 1 << precision);
    assert!(sketch.registers.iter().all(|&register| register == 0));
    Ok(())
}

#[rstest]
#[case::first_bit_after_index_is_rank_one(&[hash_for(4, 3, 1)], 3, 1)]
#[case::three_leading_zeros_is_rank_four(&[hash_for(4, 5, 4)], 5, 4)]
#[case::all_zero_hash_takes_the_largest_rank(&[0], 0, 61)]
#[case::all_ones_hash_is_last_register_rank_one(&[u64::MAX], 15, 1)]
#[case::register_keeps_the_largest_rank(&[hash_for(4, 2, 5), hash_for(4, 2, 2)], 2, 5)]
fn test_insert_hash_updates_register(
    #[case] hashes: &[u64],
    #[case] index: usize,
    #[case] expected_rank: u8,
) -> VortexResult<()> {
    let mut sketch = HyperLogLog::try_new(MIN_PRECISION)?;
    for &hash in hashes {
        sketch.insert_hash(hash);
    }
    assert_eq!(sketch.registers[index], expected_rank);
    Ok(())
}

#[rstest]
#[case::rank_one(1)]
#[case::middle_rank(30)]
#[case::largest_rank(u32::MAX)]
fn test_one_occupied_register_estimates_one(#[case] rank: u32) -> VortexResult<()> {
    let mut sketch = HyperLogLog::try_new(PRECISION)?;
    sketch.insert_hash(hash_for(PRECISION, 7, rank));
    assert_eq!(sketch.estimate(), 1);
    Ok(())
}

#[test]
fn test_estimate_extremes() -> VortexResult<()> {
    assert_eq!(HyperLogLog::try_new(PRECISION)?.estimate(), 0);

    // An infinite estimate saturates.
    let mut saturated = HyperLogLog::try_new(MIN_PRECISION)?;
    for index in 0..(1u64 << MIN_PRECISION) {
        saturated.insert_hash(hash_for(MIN_PRECISION, index, u32::MAX));
    }
    assert_eq!(saturated.estimate(), u64::MAX);
    Ok(())
}

#[test]
fn test_duplicates_do_not_change_the_sketch() -> VortexResult<()> {
    let mut sketch = sketch_of(PRECISION, 0..100)?;
    let before = sketch.clone();
    for value in 0..100i64 {
        sketch.insert(&value.to_le_bytes());
    }
    assert_eq!(sketch, before);
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[rstest]
fn test_estimate_accuracy(
    #[values(MIN_PRECISION, 8, PRECISION, 14, MAX_PRECISION)] precision: u8,
    #[values(10, 1_000, 100_000)] distinct_count: i64,
) -> VortexResult<()> {
    let sketch = sketch_of(precision, 0..distinct_count)?;
    assert_within_three_standard_errors(
        sketch.estimate(),
        distinct_count.unsigned_abs(),
        precision,
    );
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_estimate_accuracy_at_a_million() -> VortexResult<()> {
    let sketch = sketch_of(PRECISION, 0..1_000_000)?;
    assert_within_three_standard_errors(sketch.estimate(), 1_000_000, PRECISION);
    Ok(())
}

#[test]
fn test_merge_equals_the_sketch_of_the_union() -> VortexResult<()> {
    let first = sketch_of(PRECISION, 0..700)?;
    let second = sketch_of(PRECISION, 300..1_000)?;
    let union = sketch_of(PRECISION, 0..1_000)?;

    let mut first_then_second = first.clone();
    first_then_second.merge(&second)?;
    assert_eq!(first_then_second, union);

    let mut second_then_first = second;
    second_then_first.merge(&first)?;
    assert_eq!(second_then_first, union, "merge is commutative");

    let mut with_itself = first.clone();
    with_itself.merge(&first)?;
    assert_eq!(with_itself, first, "merge is idempotent");

    let mut with_empty = first.clone();
    with_empty.merge(&HyperLogLog::try_new(PRECISION)?)?;
    assert_eq!(with_empty, first, "the empty sketch is the identity");
    Ok(())
}

#[test]
fn test_merge_rejects_different_precisions() -> VortexResult<()> {
    let mut sketch = HyperLogLog::try_new(PRECISION)?;
    assert!(sketch.merge(&HyperLogLog::try_new(PRECISION + 1)?).is_err());
    Ok(())
}

#[test]
fn test_serialize_sparse_layout() -> VortexResult<()> {
    let mut sketch = HyperLogLog::try_new(PRECISION)?;
    assert_eq!(sketch.serialize(), [1, PRECISION], "empty is a bare header");

    sketch.insert_hash(hash_for(PRECISION, 258, 3));
    // Encoding, precision, then register 258 as a little-endian u32 and its rank.
    assert_eq!(sketch.serialize(), [1, PRECISION, 2, 1, 0, 0, 3]);
    Ok(())
}

#[test]
fn test_serialize_switches_to_dense_once_sparse_is_larger() -> VortexResult<()> {
    let register_count = 1usize << PRECISION;
    // Sparse costs 5 bytes per occupied register and stays while it is under `2^p` bytes.
    let largest_sparse = (register_count - 1) / 5;
    let mut sketch = HyperLogLog::try_new(PRECISION)?;
    for index in 0..largest_sparse {
        sketch.insert_hash(hash_for(PRECISION, index as u64, 1));
    }
    let sparse = sketch.serialize();
    assert_eq!((sparse[0], sparse.len()), (1, 2 + 5 * largest_sparse));

    sketch.insert_hash(hash_for(PRECISION, largest_sparse as u64, 1));
    let dense = sketch.serialize();
    assert_eq!((dense[0], dense.len()), (0, 2 + register_count));
    Ok(())
}

#[rstest]
#[case::empty(0)]
#[case::sparse(10)]
#[case::dense(100_000)]
fn test_serialization_roundtrip(#[case] distinct_count: i64) -> VortexResult<()> {
    let sketch = sketch_of(PRECISION, 0..distinct_count)?;
    assert_eq!(HyperLogLog::deserialize(&sketch.serialize())?, sketch);
    Ok(())
}

#[test]
fn test_deserialize_accepts_the_largest_rank() -> VortexResult<()> {
    let largest_rank = 64 - MIN_PRECISION + 1;
    let sketch = HyperLogLog::deserialize(&[1, MIN_PRECISION, 0, 0, 0, 0, largest_rank])?;
    assert_eq!(sketch.registers[0], largest_rank);
    Ok(())
}

#[rstest]
#[case::empty(vec![])]
#[case::missing_precision(vec![0])]
#[case::unknown_encoding(vec![2, PRECISION])]
#[case::precision_too_small(vec![1, MIN_PRECISION - 1])]
#[case::precision_too_large(vec![1, MAX_PRECISION + 1])]
#[case::dense_too_short(vec![0, MIN_PRECISION, 0])]
#[case::dense_too_long([vec![0, MIN_PRECISION], vec![0; 17]].concat())]
#[case::dense_rank_above_maximum([vec![0, MIN_PRECISION], vec![0; 15], vec![62]].concat())]
#[case::sparse_trailing_bytes(vec![1, MIN_PRECISION, 0, 0, 0])]
#[case::sparse_index_out_of_range(vec![1, MIN_PRECISION, 16, 0, 0, 0, 1])]
#[case::sparse_rank_zero(vec![1, MIN_PRECISION, 0, 0, 0, 0, 0])]
#[case::sparse_rank_above_maximum(vec![1, MIN_PRECISION, 0, 0, 0, 0, 62])]
#[case::sparse_duplicate_index(vec![1, MIN_PRECISION, 3, 0, 0, 0, 1, 3, 0, 0, 0, 2])]
#[case::sparse_decreasing_index(vec![1, MIN_PRECISION, 4, 0, 0, 0, 1, 3, 0, 0, 0, 1])]
fn test_deserialize_rejects_malformed(#[case] bytes: Vec<u8>) {
    assert!(
        HyperLogLog::deserialize(&bytes).is_err(),
        "expected a malformed sketch to be rejected: {bytes:?}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_serialized_bytes_golden() -> VortexResult<()> {
    // Changing these bytes breaks merging with sketches already written to files.
    check_metadata(
        "approx_distinct_hll_p12_sparse.partial",
        &sketch_of(PRECISION, 0..20)?.serialize(),
    );
    check_metadata(
        "approx_distinct_hll_p12_dense.partial",
        &sketch_of(PRECISION, 0..10_000)?.serialize(),
    );
    Ok(())
}
