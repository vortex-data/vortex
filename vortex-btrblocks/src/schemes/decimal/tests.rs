// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::LazyLock;

use rand::RngExt;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use rstest::rstest;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DecimalArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::i256;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::DecimalBytePartsArraySlotsExt;
use vortex_decimal_byte_parts::decimal_byte_parts_v2_id;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use super::DecimalScheme;
use super::DecimalSchemeV2;
use crate::BtrBlocksCompressor;
use crate::BtrBlocksCompressorBuilder;
use crate::SchemeExt;
use crate::SchemeId;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_decimal_byte_parts::initialize(&session);
    session
});

/// Number of values per array: large enough for cascaded integer schemes to use sampling.
const N: usize = 16_384;

fn ten_pow(exp: u32) -> i256 {
    i256::from_i128(10).wrapping_pow(exp)
}

/// Deterministic 24-bit noise, so the low part of each value is neither constant nor a
/// sequence — the realistic shape for a wide decimal column with a large fixed magnitude.
fn noise(seed: u64) -> impl Iterator<Item = i128> {
    let mut rng = StdRng::seed_from_u64(seed);
    iter::repeat_with(move || i128::from(rng.random::<u32>() >> 8))
}

/// `i128`-backed values that need more than 64 bits, so the encoding must carry one lower
/// part.
fn wide_i128_array(validity: Validity) -> DecimalArray {
    let base = 10i128.pow(25);
    let values: Buffer<i128> = noise(7).take(N).map(|delta| base + delta).collect();
    DecimalArray::new(values, DecimalDType::new(38, 2), validity)
}

/// `i256`-backed values that need more than 128 bits, so the encoding must carry three
/// lower parts.
fn wide_i256_array(validity: Validity) -> DecimalArray {
    let base = ten_pow(40);
    let values: Buffer<i256> = noise(11)
        .take(N)
        .map(|delta| base + i256::from_i128(delta))
        .collect();
    DecimalArray::new(values, DecimalDType::new(76, 2), validity)
}

/// Compress with no restriction on serialized IDs, which selects the v2 scheme.
fn compress(array: &ArrayRef) -> VortexResult<ArrayRef> {
    BtrBlocksCompressor::default().compress(array, &mut SESSION.create_execution_ctx())
}

/// Compress as a writer whose editions permit the frozen byte-parts format but not v2.
fn compress_v1_only(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let v1_only = HashSet::from([DecimalByteParts.id()]);
    BtrBlocksCompressorBuilder::default()
        .allow_serialized_ids(&v1_only)
        .build()
        .compress(array, &mut SESSION.create_execution_ctx())
}

fn byte_parts(array: &ArrayRef) -> &ArrayRef {
    assert!(
        array.is::<DecimalByteParts>(),
        "expected DecimalByteParts, got {}",
        array.encoding_id()
    );
    array
}

fn lower_part_count(array: &ArrayRef) -> usize {
    byte_parts(array)
        .as_opt::<DecimalByteParts>()
        .vortex_expect("byte parts array")
        .lower_parts()
        .len()
}

/// When the writer may emit the v2 format, values too wide for a single signed part split into
/// lower parts: one for `i128` storage, three for `i256`.
#[rstest]
#[case::i128(wide_i128_array(Validity::NonNullable).into_array(), 1)]
#[case::i128_nullable(wide_i128_array(Validity::from_iter((0..N).map(|i| i % 3 != 0))).into_array(), 1)]
#[case::i256(wide_i256_array(Validity::NonNullable).into_array(), 3)]
#[case::i256_nullable(wide_i256_array(Validity::from_iter((0..N).map(|i| i % 5 != 0))).into_array(), 3)]
fn test_wide_decimals_split_when_v2_is_permitted(
    #[case] array: ArrayRef,
    #[case] expected_lower_parts: usize,
    #[values(false, true)] explicit_ids: bool,
) -> VortexResult<()> {
    let mut builder = BtrBlocksCompressorBuilder::default();
    if explicit_ids {
        builder = builder.allow_serialized_ids(&HashSet::from([
            DecimalByteParts.id(),
            decimal_byte_parts_v2_id(),
        ]));
    }
    let compressed = builder
        .build()
        .compress(&array, &mut SESSION.create_execution_ctx())?;
    assert_eq!(lower_part_count(&compressed), expected_lower_parts);
    assert_eq!(compressed.dtype(), array.dtype());
    assert_arrays_eq!(array, compressed, &mut SESSION.create_execution_ctx());

    let serialization = SESSION
        .array_serialize(&compressed)?
        .vortex_expect("byte parts arrays are serializable");
    assert_eq!(serialization.serialized_id, decimal_byte_parts_v2_id());
    Ok(())
}

/// A writer that may emit only the frozen format leaves wide values as the canonical decimal:
/// splitting them would need lower parts, for which no single-part form exists.
#[rstest]
#[case::i128(wide_i128_array(Validity::NonNullable).into_array())]
#[case::i128_nullable(wide_i128_array(Validity::from_iter((0..N).map(|i| i % 3 != 0))).into_array())]
#[case::i256(wide_i256_array(Validity::NonNullable).into_array())]
#[case::i256_nullable(wide_i256_array(Validity::from_iter((0..N).map(|i| i % 5 != 0))).into_array())]
fn test_wide_decimals_stay_canonical_without_v2(#[case] array: ArrayRef) -> VortexResult<()> {
    let compressed = compress_v1_only(&array)?;

    assert!(
        compressed.as_opt::<DecimalByteParts>().is_none(),
        "expected the wide decimal to be left canonical, got {}",
        compressed.encoding_id()
    );
    assert_eq!(compressed.dtype(), array.dtype());
    assert_arrays_eq!(array, compressed, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[test]
fn test_i256_decimal_round_trips_extreme_values() -> VortexResult<()> {
    // Every 64-bit window exercised, including the sign boundary of the most significant
    // part. Bounded by the precision so the values are legal `Decimal(76, 0)` scalars.
    let max = ten_pow(76) - i256::ONE;
    let values: Buffer<i256> = (0..N)
        .map(|i| match i % 8 {
            0 => i256::ZERO,
            1 => i256::ONE,
            2 => i256::ZERO - i256::ONE,
            3 => i256::from_parts(u128::MAX, 0),
            4 => i256::from_parts(0, 1),
            5 => i256::from_parts(0, -1),
            6 => max,
            _ => i256::ZERO - max,
        })
        .collect();
    let array =
        DecimalArray::new(values, DecimalDType::new(76, 0), Validity::NonNullable).into_array();

    let compressed = compress(&array)?;
    assert_arrays_eq!(array, compressed, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[rstest]
fn test_narrow_decimal_has_no_lower_parts(
    #[values(false, true)] v1_only: bool,
) -> VortexResult<()> {
    // Values that fit 64 bits are narrowed rather than split, even when the declared
    // precision needs an i256.
    let values: Buffer<i256> = (0..N as i128).map(|i| i256::from_i128(i * 3)).collect();
    let array =
        DecimalArray::new(values, DecimalDType::new(76, 2), Validity::NonNullable).into_array();

    let compressed = if v1_only {
        compress_v1_only(&array)?
    } else {
        compress(&array)?
    };
    assert_eq!(lower_part_count(&compressed), 0);
    assert_arrays_eq!(array, compressed, &mut SESSION.create_execution_ctx());

    // Narrow values keep the frozen format even with the v2 scheme.
    let serialization = SESSION
        .array_serialize(&compressed)?
        .vortex_expect("byte parts arrays are serializable");
    assert_eq!(serialization.serialized_id, DecimalByteParts.id());
    Ok(())
}

#[rstest]
fn test_narrow_precision_with_wide_null_slot(
    #[values(false, true)] v1_only: bool,
) -> VortexResult<()> {
    let array = DecimalArray::new(
        buffer![1i64, i64::MAX, 3],
        DecimalDType::new(2, 0),
        Validity::from_iter([true, false, true]),
    )
    .into_array();
    let compressed = if v1_only {
        compress_v1_only(&array)?
    } else {
        compress(&array)?
    };
    assert_arrays_eq!(array, compressed, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[rstest]
#[case::neither(vec![], false)]
#[case::v1(vec![DecimalByteParts.id()], true)]
#[case::v2_without_v1(vec![decimal_byte_parts_v2_id()], false)]
#[case::both(vec![DecimalByteParts.id(), decimal_byte_parts_v2_id()], true)]
fn test_decimal_scheme_requires_every_possible_wire_id(
    #[case] allowed: Vec<ArrayId>,
    #[case] enabled: bool,
) {
    let compressor = BtrBlocksCompressorBuilder::default()
        .allow_serialized_ids(&allowed.into_iter().collect())
        .build();
    assert_eq!(compressor.has_scheme(DecimalScheme.id()), enabled);
    assert_eq!(compressor.has_scheme(DecimalSchemeV2.id()), enabled);
}

#[rstest]
#[case::v1(DecimalScheme.id())]
#[case::v2(DecimalSchemeV2.id())]
fn test_excluding_either_decimal_version_removes_the_chain(#[case] excluded: SchemeId) {
    let compressor = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([excluded])
        .build();
    assert!(!compressor.has_scheme(DecimalScheme.id()));
    assert!(!compressor.has_scheme(DecimalSchemeV2.id()));
}

#[test]
fn test_canonical_of_compressed_wide_decimal_keeps_storage_width() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let array = wide_i128_array(Validity::NonNullable).into_array();
    let canonical = compress(&array)?.execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(canonical.values_type(), DecimalType::I128);

    let array = wide_i256_array(Validity::NonNullable).into_array();
    let canonical = compress(&array)?.execute::<DecimalArray>(&mut ctx)?;
    assert_eq!(canonical.values_type(), DecimalType::I256);
    Ok(())
}
