// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression round trips and regressions involving the built-in schemes.
//!
//! These tests cover scheme composition and configuration. Individual encoding kernels and the
//! generic selection algorithm are tested in their owning crates.

use std::f64::consts::PI;
use std::iter;
use std::sync::Arc;

use rand::Rng;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
use vortex_alp::ALP;
use vortex_alp::ALPArrayExt;
use vortex_alp::ALPArraySlotsExt;
#[cfg(feature = "zstd")]
use vortex_array::ArrayId;
#[cfg(feature = "zstd")]
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_datetime_parts::DateTimeParts;
#[cfg(feature = "zstd")]
use vortex_edition::EDITION_DECLARATIONS;
#[cfg(feature = "zstd")]
use vortex_edition::EDITION_FAMILIES;
#[cfg(feature = "zstd")]
use vortex_edition::EditionId;
#[cfg(feature = "zstd")]
use vortex_edition::EditionSessionExt;
#[cfg(feature = "zstd")]
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::Delta;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::OnPair;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

use crate::BtrBlocksCompressor;
use crate::BtrBlocksCompressorBuilder;
use crate::ChildDecision;
use crate::ChildPolicy;
use crate::ChildRequest;
use crate::SESSION;
use crate::Scheme;
#[cfg(feature = "zstd")]
use crate::schemes::binary;
use crate::schemes::binary::VarBinScheme;
use crate::schemes::boolean::BoolRunEndScheme;
use crate::schemes::float::ALPRDScheme;
use crate::schemes::float::ALPScheme;
#[cfg(feature = "pco")]
use crate::schemes::float::PcoScheme as FloatPcoScheme;
use crate::schemes::integer::BitPackingScheme;
#[cfg(feature = "pco")]
use crate::schemes::integer::PcoScheme as IntPcoScheme;
use crate::schemes::integer::RunEndScheme;
#[cfg(feature = "zstd")]
use crate::schemes::string;
use crate::schemes::string::FSSTScheme;
use crate::schemes::string::OnPairScheme;

#[track_caller]
fn assert_roundtrip(compressor: &BtrBlocksCompressor, input: &ArrayRef) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(input, &mut ctx)?;
    assert_arrays_eq!(&compressed, input, &mut ctx);

    Ok(compressed)
}

#[rstest]
#[case::runs(false)]
#[case::scattered(true)]
fn test_sparse_boolean_compression(
    #[case] scattered: bool,
    #[values(false, true)] invert: bool,
    #[values(0, 5)] offset: usize,
) -> VortexResult<()> {
    let input = BoolArray::from_iter((0..100_000).map(|i| {
        let value = if scattered {
            i % 100 == 0
        } else {
            (1000..2000).contains(&i)
        };
        value ^ invert
    }))
    .into_array();
    let input = input.slice(offset..99_997)?;
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.nbytes() < input.nbytes() / 4);
    Ok(())
}

#[rstest]
fn test_container_validity_is_compressed(
    #[values("struct", "list", "listview", "fixed_size_list")] kind: &str,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let validity = Validity::Array(
        BoolArray::from_iter((0..100_000).map(|i| (1000..2000).contains(&i))).into_array(),
    );
    let elements = PrimitiveArray::from_iter(0..100_000i32).into_array();
    let input = match kind {
        "struct" => {
            StructArray::try_new(["value"].into(), [elements], 100_000, validity)?.into_array()
        }
        "list" => ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(0..=100_000u32).into_array(),
            validity,
            &mut ctx,
        )?
        .into_array(),
        "listview" => ListViewArray::try_new(
            elements,
            PrimitiveArray::from_iter(0..100_000u32).into_array(),
            PrimitiveArray::from_iter(iter::repeat_n(1u32, 100_000)).into_array(),
            validity,
            &mut ctx,
        )?
        .into_array(),
        "fixed_size_list" => {
            FixedSizeListArray::try_new(elements, 1, validity, 100_000)?.into_array()
        }
        _ => unreachable!(),
    };
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.validity()?.to_array(input.len()).nbytes() < 100);
    Ok(())
}

#[rstest]
fn test_leaf_validity_is_compressed(
    #[values("bool", "integer", "float", "decimal", "string")] kind: &str,
) -> VortexResult<()> {
    let validity = Validity::Array(
        BoolArray::from_iter((0..100_000).map(|i| (1000..2000).contains(&i))).into_array(),
    );
    let input = match kind {
        "bool" => BoolArray::new(
            BitBuffer::from_iter((0..100_000).map(|i| i % 2 == 0)),
            validity,
        )
        .into_array(),
        "integer" => PrimitiveArray::new(Buffer::from_iter(0..100_000i64), validity).into_array(),
        "float" => PrimitiveArray::new(
            Buffer::from_iter((0..100_000).map(|i| f64::from(i) / 3.0)),
            validity,
        )
        .into_array(),
        "decimal" => DecimalArray::new(
            (0..100_000i64).collect(),
            DecimalDType::new(12, 2),
            validity,
            &mut SESSION.create_execution_ctx(),
        )
        .into_array(),
        "string" => VarBinViewArray::from_iter(
            (0..100_000).map(|i| (1000..2000).contains(&i).then(|| format!("value-{i}"))),
            DType::Utf8(Nullability::Nullable),
        )
        .into_array(),
        _ => unreachable!(),
    };
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&BoolRunEndScheme)
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.validity()?.to_array(input.len()).nbytes() < 100);
    Ok(())
}

#[test]
fn test_default_compressor_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let integers = PrimitiveArray::from_option_iter(
        (0..2048i32).map(|i| (i % 7 != 0).then_some(1_000_000 + (i * 37) % 100)),
    );
    let floats = PrimitiveArray::from_iter((0..2048).map(|i| f64::from(i) / 100.0));
    let strings = VarBinViewArray::from_iter(
        (0..2048).map(|i| (i % 7 != 0).then(|| format!("https://example.com/products/{i}"))),
        DType::Utf8(Nullability::Nullable),
    );
    let binary = VarBinViewArray::from_iter(
        (0..2048).map(|i| (i % 7 != 0).then(|| format!("PREFIX_{i:09}").into_bytes())),
        DType::Binary(Nullability::Nullable),
    );
    let decimals = DecimalArray::new(
        (0..2048i64).map(|i| i * 123).collect(),
        DecimalDType::new(12, 2),
        Validity::NonNullable,
        &mut ctx,
    );
    let timestamps = TemporalArray::new_timestamp(
        PrimitiveArray::from_iter((0..2048i64).map(|i| 1_700_000_000_000_000 + i * 123_456))
            .into_array(),
        TimeUnit::Microseconds,
        None,
    );
    let elements = StructArray::from_fields(&[
        ("integer", integers.into_array()),     //
        ("float", floats.into_array()),         //
        ("string", strings.into_array()),       //
        ("binary", binary.into_array()),        //
        ("decimal", decimals.into_array()),     //
        ("timestamp", timestamps.into_array()), //
    ])?;
    let offsets = PrimitiveArray::from_iter((0..=2048i32).step_by(4));
    let input = ListArray::try_new(
        elements.into_array(),
        offsets.into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();

    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.nbytes() < input.nbytes());

    Ok(())
}

#[rstest]
#[case::seconds(TimeUnit::Seconds)]
#[case::milliseconds(TimeUnit::Milliseconds)]
fn test_extreme_timestamps_roundtrip(#[case] unit: TimeUnit) -> VortexResult<()> {
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let input = TemporalArray::new_timestamp(
        PrimitiveArray::from_iter((0..2048i64).map(|i| match i % 3 {
            0 => i64::MAX - i,
            1 => i64::MIN + i,
            _ => 1_700_000_000 + i,
        }))
        .into_array(),
        unit,
        None,
    )
    .into_array();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.is::<DateTimeParts>());

    Ok(())
}

#[rstest]
#[case::unaligned({
    let mut rng = StdRng::seed_from_u64(7);
    let mut value = 500_000i32;
    PrimitiveArray::from_iter((0..1025).map(|_| {
        value += 1 + (rng.next_u32() % 6) as i32;
        value
    }))
})]
#[case::nullable_unaligned(
    PrimitiveArray::from_option_iter(iter::once(None).chain((1i32..=100_000).map(Some)))
)]
fn test_delta_unaligned_roundtrip(#[case] input: PrimitiveArray) -> VortexResult<()> {
    // Zero-padding the final chunk used to inflate the delta span and reject this scheme.
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let input = input.into_array();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.is::<Delta>());

    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(sum(&compressed, &mut ctx)?, sum(&input, &mut ctx)?);

    Ok(())
}

/// Series of 30-second timestamps, each with an occasional missed scrape, that restart in the
/// middle of a FastLanes chunk. Each restart leaves one large negative residual, which used to
/// inflate Delta's estimated span to the full width, although the cascade under Delta patches
/// those few outliers.
#[test]
fn test_delta_chosen_for_restarting_series() -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(7);
    let mut ts = 0i64;
    let input = PrimitiveArray::from_iter((0..131_072).map(|i| {
        if i % 30_000 == 0 {
            ts = 1_700_000_000_000;
        }
        ts += if rng.next_u32() % 1000 == 0 {
            60_000
        } else {
            30_000
        };
        ts
    }))
    .into_array();

    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.is::<Delta>());
    assert!(compressed.nbytes() * 16 < input.nbytes());

    Ok(())
}

/// Fixed-size lists of random bytes, like UUIDs, with the rows `is_valid` picks valid. The null
/// rows hold zeros, as Arrow's fixed-size binary arrays do.
fn random_byte_lists(rows: usize, is_valid: impl Fn(usize) -> bool) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(42);
    let elements = (0..rows * 16)
        .map(|i| {
            if is_valid(i / 16) {
                rng.next_u32().to_le_bytes()[0]
            } else {
                0
            }
        })
        .collect::<Buffer<u8>>();
    let validity = Validity::from_iter((0..rows).map(&is_valid));
    FixedSizeListArray::new(elements.into_array(), 16, validity, rows).into_array()
}

#[rstest]
#[case::mostly_null(|row| row % 20 == 0, true)]
#[case::half_null(|row| row % 2 == 0, true)]
// Dropping one row in twenty saves less than storing the valid rows' positions costs.
#[case::few_nulls(|row| row % 20 != 0, false)]
#[case::all_valid(|_| true, false)]
fn test_fixed_size_list_drops_null_rows(
    #[case] is_valid: fn(usize) -> bool,
    #[case] sparse: bool,
) -> VortexResult<()> {
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let input = random_byte_lists(8_000, is_valid);
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert_eq!(
        compressed.is::<Sparse>(),
        sparse,
        "{}",
        compressed.display_tree()
    );

    Ok(())
}

#[test]
fn test_fixed_size_list_prefers_dense_when_elements_compress_better_than_sparse() -> VortexResult<()>
{
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let rows = 8_000;
    let list_size: u32 = 16;

    // 50% pseudo-random nulls with constant elements: dense FixedSizeList compresses elements to a
    // constant + 1-bit/row validity (~1 KiB), far smaller than Sparse with ~4,000 random indices.
    let mut rng = StdRng::seed_from_u64(123);
    let validity = Validity::from_iter((0..rows).map(|_| rng.next_u32() % 2 == 0));
    let elements = Buffer::from_iter(iter::repeat_n(42u64, rows * list_size as usize)).into_array();
    let dense_friendly = FixedSizeListArray::new(elements, list_size, validity, rows).into_array();
    let compressed_dense = assert_roundtrip(&compressor, &dense_friendly)?;
    assert!(
        compressed_dense.is::<FixedSizeList>(),
        "expected dense FixedSizeList over Sparse:\n{}",
        compressed_dense.display_tree()
    );

    // 95% nulls with random elements still pick Sparse because dense elements cannot shrink.
    let mut sparse_rng = StdRng::seed_from_u64(456);
    let valid_flags: Vec<bool> = (0..rows).map(|_| sparse_rng.next_u32() % 20 == 0).collect();
    let sparse_friendly = random_byte_lists(rows, |row| valid_flags[row]);
    let compressed_sparse = assert_roundtrip(&compressor, &sparse_friendly)?;
    assert!(
        compressed_sparse.is::<Sparse>(),
        "expected Sparse for 95%-null random FixedSizeList:\n{}",
        compressed_sparse.display_tree()
    );

    Ok(())
}

#[cfg(feature = "zstd")]
#[rstest]
#[case::array_level(CORE_2026_08_3, vortex_zstd::Zstd.id())]
#[case::buffer_level(vortex_zstd::editions::ZSTD_2026_02, vortex_zstd::ZstdBuffers.id())]
fn test_cuda_binary_zstd_follows_editions(
    #[case] edition: EditionId,
    #[case] expected: ArrayId,
) -> VortexResult<()> {
    let values: Vec<_> = (0..1024u32)
        .map(|i| {
            let mut value = Vec::from(&b"common binary payload prefix "[..]);
            value.extend_from_slice(&i.to_le_bytes());
            value.extend_from_slice(&[b'x'; 96]);
            value
        })
        .collect();
    let input = VarBinViewArray::from_iter_bin(values.iter().map(Vec::as_slice)).into_array();

    // The CUDA preset enables both Zstd schemes; the edition filter must choose between them.
    let session = VortexSession::empty();
    vortex_zstd::initialize(&session);
    for family in EDITION_FAMILIES {
        session.editions().declare_family(family)?;
    }
    for declaration in EDITION_DECLARATIONS {
        session.register_edition(declaration)?;
    }
    session.enable_edition(edition)?;
    let compressor = BtrBlocksCompressorBuilder::from_session(&session)
        .only_cuda_compatible()
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert_eq!(compressed.encoding_id(), expected);

    Ok(())
}

static BITPACKING_V1: BitPackingScheme = BitPackingScheme::v1();
static BITPACKING_V2: BitPackingScheme = BitPackingScheme::v2();

/// A scheme that stores its input's validity registers it, so it is compressed.
#[rstest]
#[case::bitpacking_v1(&BITPACKING_V1)]
#[case::bitpacking_v2(&BITPACKING_V2)]
#[cfg_attr(feature = "pco", case::pco(&IntPcoScheme))]
fn test_integer_scheme_compresses_its_validity(
    #[case] scheme: &'static dyn Scheme,
) -> VortexResult<()> {
    let input = PrimitiveArray::from_option_iter(
        (0..100_000i32).map(|i| (!(1000..2000).contains(&i)).then_some(i % 100)),
    )
    .into_array();
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(scheme)
        .with_new_scheme(&BoolRunEndScheme)
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;

    assert!(!compressed.is_canonical(), "{}", compressed.encoding_id());
    assert!(compressed.validity()?.to_array(input.len()).nbytes() < 100);
    Ok(())
}

#[rstest]
#[case::saves_one_bit(64, false)]
#[case::saves_four_bits(8, true)]
fn test_bitpacking_skips_small_savings(
    #[case] min_value: u8,
    #[case] expect_packed: bool,
) -> VortexResult<()> {
    let input = PrimitiveArray::from_iter((0..7u8).cycle().take(10_000).map(|v| min_value + v))
        .into_array();
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&BITPACKING_V2)
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;

    assert_eq!(
        compressed.is::<BitPacked>(),
        expect_packed,
        "{}",
        compressed.encoding_id()
    );
    assert_eq!(
        compressed.is::<Primitive>(),
        !expect_packed,
        "{}",
        compressed.encoding_id()
    );
    Ok(())
}

/// A float scheme that stores its input's validity registers it, so it is compressed.
#[rstest]
#[case::alprd(&ALPRDScheme)]
#[cfg_attr(feature = "pco", case::pco(&FloatPcoScheme))]
fn test_float_scheme_compresses_its_validity(
    #[case] scheme: &'static dyn Scheme,
) -> VortexResult<()> {
    let input = PrimitiveArray::from_option_iter(
        (0..100_000).map(|i| (!(1000..2000).contains(&i)).then(|| f64::from(i).sqrt())),
    )
    .into_array();
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(scheme)
        .with_new_scheme(&BoolRunEndScheme)
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;

    assert!(!compressed.is_canonical(), "{}", compressed.encoding_id());
    assert!(compressed.validity()?.to_array(input.len()).nbytes() < 100);
    Ok(())
}

/// ALP stores the values it cannot encode as patches, which it keeps as they are.
#[test]
fn test_alp_keeps_its_patches() -> VortexResult<()> {
    let input = PrimitiveArray::from_iter((0..100_000).map(|i| {
        if i % 1000 == 0 {
            PI
        } else {
            f64::from(i % 1000) / 10.0
        }
    }))
    .into_array();
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&ALPScheme)
        .with_new_scheme(&BITPACKING_V1)
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;

    let alp = if use_experimental_patches() {
        compressed
            .as_opt::<Patched>()
            .ok_or_else(|| vortex_err!("expected Patched, got {}", compressed.encoding_id()))?
            .inner()
            .clone()
    } else {
        compressed
    };
    let alp = alp
        .as_opt::<ALP>()
        .ok_or_else(|| vortex_err!("expected ALP, got {}", alp.encoding_id()))?;
    assert!(
        alp.encoded().is::<BitPacked>(),
        "{}",
        alp.encoded().encoding_id()
    );
    assert_eq!(
        alp.patches().is_some(),
        !use_experimental_patches(),
        "the patches are inside ALP unless a Patched array wraps it"
    );
    Ok(())
}

static ONPAIR: OnPairScheme = OnPairScheme::new(DEFAULT_CONFIG.max_dict_bits);

/// A string or binary scheme that stores its input's validity registers it, so it is compressed.
#[rstest]
#[case::fsst(&FSSTScheme, DType::Utf8(Nullability::Nullable))]
#[case::onpair(&ONPAIR, DType::Utf8(Nullability::Nullable))]
#[cfg_attr(feature = "zstd", case::string_zstd(&string::ZstdScheme, DType::Utf8(Nullability::Nullable)))]
#[cfg_attr(
    feature = "zstd",
    case::string_zstd_buffers(&string::ZstdBuffersScheme, DType::Utf8(Nullability::Nullable))
)]
#[cfg_attr(feature = "zstd", case::binary_zstd(&binary::ZstdScheme, DType::Binary(Nullability::Nullable)))]
#[cfg_attr(
    feature = "zstd",
    case::binary_zstd_buffers(&binary::ZstdBuffersScheme, DType::Binary(Nullability::Nullable))
)]
fn test_bytes_scheme_compresses_its_validity(
    #[case] scheme: &'static dyn Scheme,
    #[case] dtype: DType,
) -> VortexResult<()> {
    let input = VarBinViewArray::from_iter(
        (0..100_000).map(|i| {
            (!(1000..2000).contains(&i)).then(|| format!("value-{}", i % 100).into_bytes())
        }),
        dtype,
    )
    .into_array();
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(scheme)
        .with_new_scheme(&BoolRunEndScheme)
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;

    assert!(!compressed.is_canonical(), "{}", compressed.encoding_id());
    assert!(compressed.validity()?.to_array(input.len()).nbytes() < 100);
    Ok(())
}

/// Zstd compacts its input's buffers, which rebuilds the array, and still stores the input's
/// validity compressed.
#[cfg(feature = "zstd")]
#[rstest]
#[case::string(&string::ZstdScheme, DType::Utf8(Nullability::Nullable))]
#[case::binary(&binary::ZstdScheme, DType::Binary(Nullability::Nullable))]
fn test_zstd_compacts_and_compresses_its_validity(
    #[case] scheme: &'static dyn Scheme,
    #[case] dtype: DType,
) -> VortexResult<()> {
    // Values longer than 12 bytes live in the data buffers, and the slice references a tenth of
    // them, so compaction rewrites the array.
    let input = VarBinViewArray::from_iter(
        (0..200_000).map(|i| {
            (!(1000..2000).contains(&i))
                .then(|| format!("an outlined value {}", i % 100).into_bytes())
        }),
        dtype,
    )
    .into_array()
    .slice(0..20_000)?;
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(scheme)
        .with_new_scheme(&BoolRunEndScheme)
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;

    assert!(!compressed.is_canonical(), "{}", compressed.encoding_id());
    assert!(compressed.validity()?.to_array(input.len()).nbytes() < 100);
    Ok(())
}

/// The VarBin scheme rebuilds the validity bitmap, which it compresses rather than writing out
/// raw.
#[test]
fn test_varbin_validity_is_compressed() -> VortexResult<()> {
    let input = VarBinViewArray::from_iter(
        (0..100_000)
            .map(|i| (!(1000..2000).contains(&i)).then(|| format!("value-{i}").into_bytes())),
        DType::Binary(Nullability::Nullable),
    )
    .into_array();
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&VarBinScheme)
        .with_new_scheme(&BoolRunEndScheme)
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;

    assert!(compressed.is::<VarBin>(), "{}", compressed.encoding_id());
    assert!(compressed.validity()?.to_array(input.len()).nbytes() < 100);
    Ok(())
}

/// Sparse stores each null as a patch, so a quarter of nulls among mostly-zero values must not
/// pick it. This is the shape of a genotype column: most samples are homozygous reference (0),
/// about a fifth have no call (null), and a few carry an alternate allele.
#[test]
fn test_sparse_counts_nulls_as_patches() -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(0);
    let len = 131_072;
    let input = PrimitiveArray::from_option_iter((0..len).map(|_| {
        let draw = rng.next_u32() % 400;
        if draw < 100 {
            None
        } else if draw < 103 {
            Some(1u8)
        } else {
            Some(0u8)
        }
    }))
    .into_array();
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;

    assert!(!compressed.is::<Sparse>(), "{}", compressed.display_tree());
    // A raw validity bitmap is len / 8 bytes; the values cost little beyond it.
    assert!(
        compressed.nbytes() < len / 8 + len / 16,
        "{}",
        compressed.display_tree()
    );
    Ok(())
}

/// Unique text cut at random offsets from a pool of prose, like the comment columns of TPC-H.
fn unique_prose(rows: usize, seed: u64) -> ArrayRef {
    const SYLLABLES: &[&str] = &[
        "ac", "count", "pack", "age", "re", "quest", "de", "pos", "it", "in", "struct", "ion",
        "the", "od", "ol", "ite", "pin", "to", "bean", "fox", "id", "ea", "pend", "enc", "ex",
        "cuse", "plate", "let", "asym", "ptote", "court", "dol", "phin", "mul", "ti", "pli", "er",
        "saut", "erne", "wart", "hog", "fret", "din", "os", "at", "tain", "mint", "sum", "as",
        "braid", "hock", "ey", "play", "fur", "ious", "ly", "care", "fill", "quick", "sly",
    ];

    let mut rng = StdRng::seed_from_u64(seed);
    let words: Vec<String> = (0..300)
        .map(|_| {
            (0..rng.random_range(1..4))
                .map(|_| SYLLABLES[rng.random_range(0..SYLLABLES.len())])
                .collect()
        })
        .collect();

    let mut pool = String::new();
    while pool.len() < 1 << 20 {
        pool.push_str(&words[rng.random_range(0..words.len())]);
        pool.push(' ');
    }

    let values = (0..rows).map(|_| {
        let len = rng.random_range(19..79);
        let start = rng.random_range(0..pool.len() - len);
        &pool[start..start + len]
    });
    VarBinViewArray::from_iter_str(values).into_array()
}

/// The sampled estimates used to flip between FSST and OnPair from chunk to chunk on unique text,
/// although OnPair is clearly smaller on every one of them at full size.
#[test]
fn test_unique_text_selects_onpair_on_every_chunk() -> VortexResult<()> {
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();

    for seed in 0..8 {
        let compressed = assert_roundtrip(&compressor, &unique_prose(16_384, seed))?;
        assert!(
            compressed.is::<OnPair>(),
            "chunk {seed} compressed to {}",
            compressed.encoding_id()
        );
    }

    Ok(())
}

/// Keeps every child a scheme registers.
#[derive(Debug)]
struct KeepEverything;

impl ChildPolicy for KeepEverything {
    fn decide(&self, _request: &ChildRequest<'_>) -> ChildDecision {
        ChildDecision::Keep
    }
}

#[rstest]
#[case::schemes_decide(false)]
#[case::policy_keeps(true)]
fn test_child_policy_overrides_schemes(#[case] keep: bool) -> VortexResult<()> {
    let input = PrimitiveArray::from_iter((0..1_000_000).map(|i| i / 100)).into_array();
    let mut builder = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&RunEndScheme)
        .with_new_scheme(&BITPACKING_V1);
    if keep {
        builder = builder.with_child_policy(Arc::new(KeepEverything));
    }
    let compressed = assert_roundtrip(&builder.build(), &input)?;

    assert!(compressed.is::<RunEnd>(), "{}", compressed.encoding_id());
    let children_canonical = compressed
        .slots()
        .iter()
        .flatten()
        .all(ArrayRef::is_canonical);
    assert_eq!(children_canonical, keep);
    Ok(())
}
