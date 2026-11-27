// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression round trips and regressions involving the built-in schemes.
//!
//! These tests cover scheme composition and configuration. Individual encoding kernels and the
//! generic selection algorithm are tested in their owning crates.

use std::iter;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use hegel::TestCase;
use hegel::generators as gs;
use rand::Rng;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
use vortex_alp::ALPRD;
use vortex_alp::ALPRDArraySlotsExt;
use vortex_array::ArrayId;
#[cfg(feature = "zstd")]
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_datetime_parts::DateTimeParts;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
#[cfg(feature = "zstd")]
use vortex_edition::EditionId;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_edition::declarations::unstable::UNSTABLE;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::Delta;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_mask::Mask;
use vortex_onpair::OnPair;
use vortex_pivco::Pivco;
use vortex_pivco::pivco_id;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

use crate::ArrayAndStats;
use crate::BtrBlocksCompressor;
use crate::BtrBlocksCompressorBuilder;
use crate::CascadingCompressor;
use crate::ChildDecision;
use crate::ChildPolicy;
use crate::ChildRequest;
use crate::CompressionSession;
use crate::CompressorContext;
use crate::SESSION;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::binary::VarBinScheme;
use crate::schemes::boolean::BoolRunEndScheme;
use crate::schemes::float::ALPRDScheme;
use crate::schemes::integer::BitPackingScheme;
use crate::schemes::integer::EliasFanoScheme;
use crate::schemes::integer::PivcoScheme;
use crate::schemes::integer::RunEndScheme;

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

/// A scheme that stores its input's validity compresses it before encoding.
#[test]
fn test_bitpacking_compresses_its_validity() -> VortexResult<()> {
    let input = PrimitiveArray::from_option_iter(
        (0..100_000i32).map(|i| (!(1000..2000).contains(&i)).then_some(i % 100)),
    )
    .into_array();
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&BITPACKING_V1)
        .with_new_scheme(&BoolRunEndScheme)
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;

    assert!(compressed.is::<BitPacked>(), "{}", compressed.encoding_id());
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

static PIVCO_PH: PivcoScheme = PivcoScheme::ph();
static PIVCO_PHA: PivcoScheme = PivcoScheme::pha();
static PIVCO_OP_PHA: PivcoScheme = PivcoScheme::op_pha();
static PIVCO_AUTO: PivcoScheme = PivcoScheme::auto();

fn check_pivco_scheme_roundtrip<T>(tc: &TestCase) -> VortexResult<()>
where
    T: NativePType + gs::Integer + 'static,
{
    let nullable = tc.draw(gs::booleans());
    let allow_for = tc.draw(gs::booleans());
    let scheme_idx = tc.draw(gs::integers::<u8>().max_value(3));
    let len = tc.draw(gs::integers::<usize>().min_value(1_024).max_value(2_048));
    let pattern: Vec<(T, bool)> = tc.draw(
        gs::vecs(gs::tuples2(gs::integers::<T>(), gs::booleans()))
            .min_size(1)
            .max_size(16),
    );
    let (values, valid): (Vec<T>, Vec<bool>) = pattern.iter().copied().cycle().take(len).unzip();

    let validity = if nullable {
        Validity::Array(BoolArray::from_iter(valid).into_array())
    } else {
        Validity::NonNullable
    };
    let input = PrimitiveArray::new(Buffer::from_iter(values), validity).into_array();

    let base_scheme: &'static PivcoScheme = match scheme_idx {
        0 => &PIVCO_AUTO,
        1 => &PIVCO_PH,
        2 => &PIVCO_PHA,
        _ => &PIVCO_OP_PHA,
    };
    let scheme: &'static dyn Scheme = if allow_for {
        base_scheme
    } else {
        base_scheme.refine(&|id: &ArrayId| *id == pivco_id())
    };
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(scheme)
        .with_new_scheme(&BoolRunEndScheme)
        .build();
    assert_roundtrip(&compressor, &input)?;
    Ok(())
}

/// Compressing any integer primitive array through `PivcoScheme` (any integer ptype, nullable or
/// non-nullable including arbitrary null-slot values, `FoR` on or off, and any preset mode) must
/// round-trip identically.
#[hegel::test(test_cases = 50, print_blob = true)]
fn test_pivco_scheme_roundtrip_property(tc: TestCase) {
    match tc.draw(gs::integers::<u8>().max_value(7)) {
        0 => check_pivco_scheme_roundtrip::<u8>(&tc),
        1 => check_pivco_scheme_roundtrip::<u16>(&tc),
        2 => check_pivco_scheme_roundtrip::<u32>(&tc),
        3 => check_pivco_scheme_roundtrip::<u64>(&tc),
        4 => check_pivco_scheme_roundtrip::<i8>(&tc),
        5 => check_pivco_scheme_roundtrip::<i16>(&tc),
        6 => check_pivco_scheme_roundtrip::<i32>(&tc),
        _ => check_pivco_scheme_roundtrip::<i64>(&tc),
    }
    .vortex_expect("PivcoScheme round-trip property");
}

#[rstest]
#[case::auto(&PIVCO_AUTO, false)]
#[case::ph(&PIVCO_PH, false)]
#[case::pha(&PIVCO_PHA, false)]
#[case::op_pha(&PIVCO_OP_PHA, true)]
fn test_pivco_scheme_presets_encode_pivco(
    #[case] scheme: &'static PivcoScheme,
    #[case] expect_op: bool,
) -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(42);
    let input = PrimitiveArray::from_iter((0..4_096).map(|_| {
        let u = rng.random_range(0..1000u32);
        if u < 900 {
            0u64
        } else if u < 980 {
            1
        } else {
            rng.random_range(2..4_096)
        }
    }))
    .into_array();

    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(scheme)
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    let pivco = compressed
        .as_opt::<Pivco>()
        .vortex_expect("expected Pivco on skewed u64 input");
    if expect_op {
        assert!(pivco.is_order_preserving());
    }
    Ok(())
}

#[test]
fn test_pivco_scheme_ignores_dirty_negative_null_slots() -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(271828);
    let len = 16_384usize;
    let mut values = Vec::with_capacity(len);
    let mut valid = Vec::with_capacity(len);
    for i in 0..len {
        if i % 13 == 0 {
            valid.push(false);
            // Poison masked-off null physical slots with extreme negative values.
            values.push(if i % 26 == 0 { i64::MIN } else { -999_999 });
        } else {
            valid.push(true);
            let u = rng.random_range(0..1000u32);
            let v = if u < 900 {
                0i64
            } else if u < 980 {
                1
            } else {
                rng.random_range(2..64)
            };
            values.push(v);
        }
    }
    let input = PrimitiveArray::new(
        Buffer::from_iter(values),
        Validity::Array(BoolArray::from_iter(valid).into_array()),
    )
    .into_array();

    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&PIVCO_AUTO)
        .with_new_scheme(&BoolRunEndScheme)
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(
        compressed.is::<Pivco>(),
        "expected unbiased Pivco when only null slots are negative, got {}",
        compressed.encoding_id()
    );
    Ok(())
}

#[test]
fn test_pivco_for_cascade_and_op_pha_promotion() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(2026_1126);
    let len = 16_384usize;

    // 1. Shifted positive `u64` (`1_700_000_000_000 + skewed_delta`): `FoR(Pivco)` beats both
    //    unbiased `Pivco` and `FoR(BitPacked)`.
    let shifted_u64 = PrimitiveArray::from_iter((0..len).map(|_| {
        let u = rng.random_range(0..1000u32);
        let delta = if u < 900 {
            0u64
        } else if u < 980 {
            1
        } else {
            rng.random_range(2..4_096)
        };
        1_700_000_000_000u64 + delta
    }))
    .into_array();

    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&PIVCO_AUTO)
        .build();
    let compressed_u64 = assert_roundtrip(&compressor, &shifted_u64)?;
    let for_u64 = compressed_u64
        .as_opt::<FoR>()
        .vortex_expect("expected FoR(Pivco) on shifted skewed u64");
    let child_pivco = for_u64
        .encoded()
        .as_opt::<Pivco>()
        .vortex_expect("expected Pivco child under FoR");
    // Monotone-decreasing skewed distribution promotes to `OrderPreserving` in default auto mode!
    assert!(
        child_pivco.is_order_preserving(),
        "expected default auto PivcoScheme to promote low-penalty histogram to OrderPreserving"
    );

    // Exercise `scalar_at`, `slice`, `take`, `filter`, and `compare` on `FoR(Pivco)`.
    assert_eq!(
        compressed_u64.execute_scalar(1_234, &mut ctx)?,
        shifted_u64.execute_scalar(1_234, &mut ctx)?
    );
    assert_arrays_eq!(
        &compressed_u64.slice(500..9_000)?,
        &shifted_u64.slice(500..9_000)?,
        &mut ctx
    );
    let take_idx = PrimitiveArray::from_iter([0u32, 7, 4_095, 4_096, 12_345]).into_array();
    assert_arrays_eq!(
        &compressed_u64.take(take_idx.clone())?,
        &shifted_u64.take(take_idx)?,
        &mut ctx
    );
    let mask = Mask::from_buffer(BitBuffer::from_iter((0..len).map(|i| i % 97 == 1)));
    assert_arrays_eq!(
        &compressed_u64.filter(mask.clone())?,
        &shifted_u64.filter(mask)?,
        &mut ctx
    );
    let rhs = ConstantArray::new(Scalar::from(1_700_000_000_001u64), len).into_array();
    assert_arrays_eq!(
        &compressed_u64.compare(rhs.clone(), CompareOperator::Lt)?,
        &shifted_u64.compare(rhs, CompareOperator::Lt)?,
        &mut ctx
    );

    // 2. Negative-min bounded-range nullable `i32` (`-5_000 + skewed_delta`) -> `FoR(Pivco)`.
    let shifted_neg = PrimitiveArray::from_option_iter((0..len).map(|i| {
        if i % 19 == 0 {
            None
        } else {
            let u = rng.random_range(0..1000u32);
            let delta = if u < 880 {
                0i32
            } else if u < 970 {
                1
            } else {
                rng.random_range(2..2_048)
            };
            Some(-5_000 + delta)
        }
    }))
    .into_array();
    let compressed_neg = assert_roundtrip(&compressor, &shifted_neg)?;
    let for_neg = compressed_neg
        .as_opt::<FoR>()
        .vortex_expect("expected FoR(Pivco) on negative-min bounded-range i32");
    assert!(for_neg.encoded().is::<Pivco>());

    Ok(())
}

#[rstest]
#[case::non_nullable(false)]
#[case::nullable(true)]
fn test_alprd_left_parts_cascades_to_pivco_and_cuda_keeps_bitpacked(
    #[case] nullable: bool,
) -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(2026_1127);
    let len = 16_384usize;

    // Skewed-exponent binary f32s in `[0, 1)`: full 23-bit mantissas force ALPRD, while the
    // exponent prefixes follow a geometric(0.5) distribution (~2.0 bits of entropy vs 3 bits in
    // `BitPacked`), so `left_parts` cascades to `Pivco` when `vortex.pivco` is permitted.
    let values: Vec<f32> = (0..len).map(|_| rng.random::<f32>()).collect();
    let input = if nullable {
        PrimitiveArray::from_option_iter(
            values
                .iter()
                .copied()
                .enumerate()
                .map(|(i, v)| (i % 29 != 0).then_some(v)),
        )
        .into_array()
    } else {
        PrimitiveArray::from_iter(values).into_array()
    };

    let make_session = |enable_unstable: bool| -> VortexResult<VortexSession> {
        let session = VortexSession::empty();
        vortex_alp::initialize(&session);
        vortex_fastlanes::initialize(&session);
        vortex_pivco::initialize(&session);
        for family in EDITION_FAMILIES {
            session.editions().declare_family(family)?;
        }
        for declaration in EDITION_DECLARATIONS {
            session.register_edition(declaration)?;
        }
        session.enable_edition(CORE_2026_08_3)?;
        if enable_unstable {
            session.enable_edition(UNSTABLE)?;
        }
        Ok(session)
    };

    // Core edition only: ALPRD keeps BitPacked left_parts.
    let core_session = make_session(false)?;
    let core_compressor = BtrBlocksCompressorBuilder::from_session(&core_session).build();
    let core_compressed = assert_roundtrip(&core_compressor, &input)?;
    let core_alprd = core_compressed
        .as_opt::<ALPRD>()
        .vortex_expect("expected ALPRD under core edition");
    assert!(
        core_alprd.left_parts().is::<BitPacked>(),
        "expected BitPacked left_parts under core edition, got {}",
        core_alprd.left_parts().encoding_id()
    );

    // Core + unstable: ALPRD cascades left_parts into Pivco and shrinks strictly.
    let unstable_session = make_session(true)?;
    let unstable_compressor = BtrBlocksCompressorBuilder::from_session(&unstable_session).build();
    let pivco_compressed = assert_roundtrip(&unstable_compressor, &input)?;
    let pivco_alprd = pivco_compressed
        .as_opt::<ALPRD>()
        .vortex_expect("expected ALPRD under unstable edition");
    assert!(
        pivco_alprd.left_parts().is::<Pivco>(),
        "expected Pivco left_parts under unstable edition, got {}",
        pivco_alprd.left_parts().encoding_id()
    );
    assert!(
        pivco_alprd.right_parts().is::<BitPacked>(),
        "expected BitPacked right_parts, got {}",
        pivco_alprd.right_parts().encoding_id()
    );
    assert!(
        pivco_compressed.nbytes() < core_compressed.nbytes(),
        "expected ALPRD(Pivco) ({} B) to be strictly smaller than ALPRD(BitPacked) ({} B)",
        pivco_compressed.nbytes(),
        core_compressed.nbytes()
    );

    // Verify scalar_at and slice on ALPRD(Pivco).
    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(
        pivco_compressed.execute_scalar(1_234, &mut ctx)?,
        input.execute_scalar(1_234, &mut ctx)?
    );
    assert_arrays_eq!(
        &pivco_compressed.slice(500..9_000)?,
        &input.slice(500..9_000)?,
        &mut ctx
    );

    // CUDA preset: excludes Pivco even when unstable is enabled, keeping BitPacked left_parts.
    let cuda_compressor = BtrBlocksCompressorBuilder::from_session(&unstable_session)
        .only_cuda_compatible()
        .build();
    let cuda_compressed = assert_roundtrip(&cuda_compressor, &input)?;
    let cuda_alprd = cuda_compressed
        .as_opt::<ALPRD>()
        .vortex_expect("expected ALPRD under CUDA preset");
    assert!(
        cuda_alprd.left_parts().is::<BitPacked>(),
        "expected CUDA preset to keep BitPacked left_parts, got {}",
        cuda_alprd.left_parts().encoding_id()
    );

    Ok(())
}

#[rstest]
#[case::default_compresses_validity(false)]
#[case::keep_everything_preserves_canonical_validity(true)]
fn test_pivco_honors_child_policy_on_validity(#[case] keep: bool) -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(99);
    let input = PrimitiveArray::from_option_iter((0..16_384).map(|i| {
        if (1000..2000).contains(&i) {
            None
        } else {
            let u = rng.random_range(0..1000u32);
            let delta = if u < 900 {
                0u64
            } else if u < 980 {
                1
            } else {
                rng.random_range(2..2_048)
            };
            Some(1_700_000_000_000u64 + delta)
        }
    }))
    .into_array();

    let mut builder = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&PIVCO_AUTO)
        .with_new_scheme(&BoolRunEndScheme);
    if keep {
        builder = builder.with_child_policy(Arc::new(KeepEverything));
    }
    let compressed = assert_roundtrip(&builder.build(), &input)?;
    assert!(compressed.is::<FoR>(), "{}", compressed.encoding_id());
    let validity_array = compressed.validity()?.to_array(input.len());
    assert_eq!(validity_array.is_canonical(), keep);
    Ok(())
}

/// Dummy integer scheme that panics if `expected_compression_ratio` or `compress` is ever called;
/// used to prove `ALPRDScheme` child 0's allow-list blocks arbitrary future integer schemes.
#[derive(Debug)]
struct ForbiddenFutureIntScheme;

impl Scheme for ForbiddenFutureIntScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.test.forbidden_future_int"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        panic!("ForbiddenFutureIntScheme must never be evaluated on ALPRD child 0");
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        panic!("ForbiddenFutureIntScheme must never compress ALPRD child 0");
    }
}

#[test]
fn test_alprd_child_zero_allows_only_pivco() -> VortexResult<()> {
    static PROBE_RAN: AtomicBool = AtomicBool::new(false);

    /// Matches boolean arrays so it always runs once with a real root `CompressorContext` while
    /// every integer/float/string/binary/decimal/temporal/FSL scheme is registered.
    #[derive(Debug)]
    struct BoolProbeCheck;

    impl Scheme for BoolProbeCheck {
        fn scheme_name(&self) -> &'static str {
            "vortex.test.bool_probe_alprd_child_zero"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            canonical.dtype().is_boolean()
        }

        fn produced_encodings(&self) -> Vec<ArrayId> {
            Vec::new()
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
        }

        fn compress(
            &self,
            compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            let leaf_ctx = compress_ctx.clone().as_leaf();
            assert!(!compressor.is_child_scheme_eligible(
                PIVCO_AUTO.id(),
                &leaf_ctx,
                ALPRDScheme.id(),
                0,
            ));
            let eligible_ids: Vec<_> = compressor
                .allowed_schemes()
                .iter()
                .filter(|s| {
                    compressor.is_child_scheme_eligible(s.id(), &compress_ctx, ALPRDScheme.id(), 0)
                })
                .map(|s| s.id())
                .collect();
            assert_eq!(eligible_ids, vec![PIVCO_AUTO.id()]);
            PROBE_RAN.store(true, Ordering::Relaxed);
            Ok(data.array().clone())
        }
    }

    let mut ctx = SESSION.create_execution_ctx();
    let mut schemes_for_probe: Vec<&'static dyn Scheme> = CompressionSession::default()
        .schemes()
        .into_iter()
        .filter(|s| {
            // Keep every non-bool scheme (all integer, float, string, binary, decimal, temporal,
            // FSL) so `BoolProbeCheck` is the only bool scheme and always runs.
            !matches!(s.scheme_name(), "vortex.bool.runend" | "vortex.bool.sparse")
        })
        .collect();
    schemes_for_probe.push(&EliasFanoScheme);
    schemes_for_probe.push(&ForbiddenFutureIntScheme);
    schemes_for_probe.push(&BoolProbeCheck);

    let probe_compressor = CascadingCompressor::new(schemes_for_probe);
    let bool_probe = BoolArray::from_iter([true, false, true, false]).into_array();
    drop(probe_compressor.compress(&bool_probe, &mut ctx)?);
    assert!(
        PROBE_RAN.load(Ordering::Relaxed),
        "BoolProbeCheck did not run"
    );

    // End-to-end: when `PivcoScheme` is NOT enabled, `ALPRDScheme` skips `compress_child`
    // completely and keeps `BitPacked` `left_parts` without invoking `ForbiddenFutureIntScheme`.
    let mut rng = StdRng::seed_from_u64(2026_1128);
    let input = PrimitiveArray::from_iter((0..4096).map(|_| rng.random::<f32>())).into_array();
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&ALPRDScheme)
        .with_new_scheme(&EliasFanoScheme)
        .with_new_scheme(&ForbiddenFutureIntScheme)
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    let alprd = compressed
        .as_opt::<ALPRD>()
        .vortex_expect("expected ALPRD output");
    assert!(alprd.left_parts().is::<BitPacked>());
    Ok(())
}

#[rstest]
#[case::with_for(true)]
#[case::without_for(false)]
fn test_pivco_direct_compress_negative_without_prior_estimate(
    #[case] allow_for: bool,
) -> VortexResult<()> {
    /// Forces `PivcoScheme::compress` to be called directly without ever invoking
    /// `PivcoScheme::expected_compression_ratio`.
    #[derive(Debug)]
    struct DirectPivcoCompress {
        allow_for: bool,
    }

    impl Scheme for DirectPivcoCompress {
        fn scheme_name(&self) -> &'static str {
            "vortex.test.direct_pivco_compress"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            canonical.dtype().is_int()
        }

        fn produced_encodings(&self) -> Vec<ArrayId> {
            Vec::new()
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
        }

        fn compress(
            &self,
            compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            compress_ctx: CompressorContext,
            exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            let scheme: &dyn Scheme = if self.allow_for {
                &PIVCO_AUTO
            } else {
                PIVCO_AUTO.refine(&|id: &ArrayId| *id == pivco_id())
            };
            scheme.compress(compressor, data, compress_ctx, exec_ctx)
        }
    }

    static DIRECT_WITH_FOR: DirectPivcoCompress = DirectPivcoCompress { allow_for: true };
    static DIRECT_NO_FOR: DirectPivcoCompress = DirectPivcoCompress { allow_for: false };

    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(31415);
    let input = PrimitiveArray::from_iter((0..4096).map(|_| {
        let u = rng.random_range(0..1000u32);
        let delta = if u < 900 {
            0i32
        } else if u < 980 {
            1
        } else {
            rng.random_range(2..512)
        };
        -5_000 + delta
    }))
    .into_array();
    let wide =
        PrimitiveArray::from_iter((0..2048).map(|i| if i % 2 == 0 { i64::MIN } else { i64::MAX }))
            .into_array();

    let wrapper: &'static dyn Scheme = if allow_for {
        &DIRECT_WITH_FOR
    } else {
        &DIRECT_NO_FOR
    };
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(wrapper)
        .build();

    let compressed = assert_roundtrip(&compressor, &input)?;
    if allow_for {
        let for_arr = compressed
            .as_opt::<FoR>()
            .vortex_expect("expected FoR(Pivco) when allow_for=true");
        assert!(for_arr.encoded().is::<Pivco>());
    } else {
        assert!(
            compressed.is_canonical(),
            "expected canonical fallback when allow_for=false on negative input, got {}",
            compressed.encoding_id()
        );
    }

    // Wide-span negative (`i64::MIN..=i64::MAX`, span > i64::MAX) must also never bail.
    let wide_out = compressor.compress(&wide, &mut ctx)?;
    assert_arrays_eq!(&wide_out, &wide, &mut ctx);
    assert!(wide_out.is_canonical());

    Ok(())
}
