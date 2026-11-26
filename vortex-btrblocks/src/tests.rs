// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression round trips and regressions involving the built-in schemes.
//!
//! These tests cover scheme composition and configuration. Individual encoding kernels and the
//! generic selection algorithm are tested in their owning crates.

use std::iter;
use std::sync::Arc;

use parking_lot::Mutex;
use rand::Rng;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
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
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::Delta;
use vortex_onpair::OnPair;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

use crate::BtrBlocksCompressor;
use crate::BtrBlocksCompressorBuilder;
use crate::ChildDecision;
use crate::ChildRequest;
use crate::CompressionPolicy;
use crate::MAX_CASCADE;
use crate::Plan;
use crate::SESSION;
use crate::Scheme;
use crate::SchemeExt;
use crate::SchemeId;
use crate::SchemeSelection;
use crate::SiteOutcome;
use crate::SiteRequest;
use crate::schemes::binary::VarBinScheme;
use crate::schemes::boolean::BoolRunEndScheme;
use crate::schemes::integer::BitPackingScheme;
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

impl CompressionPolicy for KeepEverything {
    fn decide_child(&self, _request: &ChildRequest<'_>) -> ChildDecision {
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
        builder = builder.with_policy(Arc::new(KeepEverything));
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

/// A site's cascade history and the scheme applied there.
type Applied = (Vec<(SchemeId, usize)>, Option<SchemeId>);

/// Answers every site with one selection, and records each site's applied scheme by its cascade
/// history.
#[derive(Debug)]
struct SelectingPolicy {
    selection: SchemeSelection,
    applied: Mutex<Vec<Applied>>,
    candidates: Mutex<Vec<Applied>>,
}

impl SelectingPolicy {
    fn new(selection: SchemeSelection) -> Arc<Self> {
        Arc::new(Self {
            selection,
            applied: Mutex::new(Vec::new()),
            candidates: Mutex::new(Vec::new()),
        })
    }

    /// The scheme applied at the root of the last compressed array.
    fn root(&self) -> Option<SchemeId> {
        self.applied
            .lock()
            .iter()
            .rev()
            .find(|(path, _)| path.is_empty())
            .and_then(|(_, scheme)| *scheme)
    }
}

impl CompressionPolicy for SelectingPolicy {
    fn select_scheme(&self, _request: &SiteRequest<'_>) -> SchemeSelection {
        self.selection.clone()
    }

    fn observe_candidate(&self, candidate: &SiteOutcome<'_>) {
        self.candidates
            .lock()
            .push((candidate.ctx.cascade_history().to_vec(), candidate.scheme));
    }

    fn observe(&self, outcome: &SiteOutcome<'_>) {
        self.applied
            .lock()
            .push((outcome.ctx.cascade_history().to_vec(), outcome.scheme));
    }
}

fn runs_compressor(policy: Option<Arc<SelectingPolicy>>) -> BtrBlocksCompressor {
    let builder = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&RunEndScheme)
        .with_new_scheme(&BITPACKING_V1);
    match policy {
        Some(policy) => builder.with_policy(policy),
        None => builder,
    }
    .build()
}

fn runs() -> ArrayRef {
    PrimitiveArray::from_iter((0..100_000).map(|i| i / 100)).into_array()
}

#[test]
fn test_policy_forces_a_scheme() -> VortexResult<()> {
    let policy = SelectingPolicy::new(SchemeSelection::Scheme(BITPACKING_V1.id()));
    let compressed = assert_roundtrip(&runs_compressor(Some(Arc::clone(&policy))), &runs())?;
    assert!(compressed.is::<BitPacked>(), "{}", compressed.encoding_id());
    assert_eq!(policy.root(), Some(BITPACKING_V1.id()));
    Ok(())
}

#[test]
fn test_policy_keeps_an_array_canonical() -> VortexResult<()> {
    let policy = SelectingPolicy::new(SchemeSelection::Canonical);
    let compressed = assert_roundtrip(&runs_compressor(Some(Arc::clone(&policy))), &runs())?;
    assert!(compressed.is_canonical(), "{}", compressed.encoding_id());
    assert_eq!(policy.root(), None);
    Ok(())
}

#[test]
fn test_policy_best_is_no_larger_than_estimates() -> VortexResult<()> {
    let estimated = assert_roundtrip(&runs_compressor(None), &runs())?;
    let policy = SelectingPolicy::new(SchemeSelection::Best(Vec::new()));
    let best = assert_roundtrip(&runs_compressor(Some(Arc::clone(&policy))), &runs())?;
    assert!(
        best.nbytes() <= estimated.nbytes(),
        "{} > {}",
        best.nbytes(),
        estimated.nbytes()
    );
    // Both root candidates are reported, winner or not.
    let roots: Vec<_> = policy
        .candidates
        .lock()
        .iter()
        .filter(|(path, _)| path.is_empty())
        .filter_map(|(_, scheme)| *scheme)
        .collect();
    assert_eq!(roots, [RunEndScheme.id(), BITPACKING_V1.id()]);
    // The children of every tried scheme are reported, under their own cascade histories.
    let applied = policy.applied.lock();
    assert!(
        applied
            .iter()
            .any(|(path, _)| path.first().is_some_and(|(s, _)| *s == RunEndScheme.id()))
    );
    Ok(())
}

#[test]
fn test_policy_ignores_an_ineligible_scheme() -> VortexResult<()> {
    let estimated = assert_roundtrip(&runs_compressor(None), &runs())?;
    let policy = SelectingPolicy::new(SchemeSelection::Scheme(VarBinScheme.id()));
    let compressed = assert_roundtrip(&runs_compressor(Some(policy)), &runs())?;
    assert_eq!(compressed.encoding_id(), estimated.encoding_id());
    Ok(())
}

#[test]
fn test_recorded_plan_replays_to_the_same_array() -> VortexResult<()> {
    let compressor = runs_compressor(None);
    let input = runs();
    let mut ctx = SESSION.create_execution_ctx();

    let (compressed, plan) = compressor.compress_recording_plan(&input, &mut ctx)?;
    assert_eq!(
        compressed.nbytes(),
        compressor.compress(&input, &mut ctx)?.nbytes()
    );
    assert_eq!(plan.at(&[]), &plan);
    assert!(matches!(plan, Plan::Scheme { .. }), "{plan}");

    let schemes: [&'static dyn Scheme; 2] = [&RunEndScheme, &BITPACKING_V1];
    let parsed = Plan::parse(&plan.to_string(), &schemes)?;
    assert_eq!(parsed, plan);

    let replayed = compressor.compress_with_plan(&input, &parsed, &mut ctx)?;
    assert_eq!(replayed.encoding_id(), compressed.encoding_id());
    assert_eq!(replayed.nbytes(), compressed.nbytes());
    assert_arrays_eq!(&replayed, &input, &mut ctx);
    Ok(())
}

#[test]
fn test_plan_forces_the_cascade() -> VortexResult<()> {
    let schemes: [&'static dyn Scheme; 2] = [&RunEndScheme, &BITPACKING_V1];
    let plan = Plan::parse(&format!("{}(canonical, _)", RunEndScheme.id()), &schemes)?;
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = runs_compressor(None).compress_with_plan(&runs(), &plan, &mut ctx)?;
    assert!(compressed.is::<RunEnd>(), "{}", compressed.encoding_id());
    let canonical: Vec<String> = compressed
        .slots()
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.as_ref().is_some_and(ArrayRef::is_canonical))
        .map(|(index, _)| compressed.slot_name(index))
        .collect();
    assert_eq!(canonical, ["values"]);
    assert_arrays_eq!(&compressed, &runs(), &mut ctx);
    Ok(())
}

#[test]
fn test_plan_rejects_unknown_schemes() {
    let schemes: [&'static dyn Scheme; 1] = [&RunEndScheme];
    assert!(Plan::parse("test.nope", &schemes).is_err());
    assert!(Plan::parse("_ _", &schemes).is_err());
}

/// Wide values with runs of 2: the default skips RunEnd below an average run of 4, while the
/// size-only model sees that one end per two 31-bit values halves the size.
fn wide_runs_of_two() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(11);
    let values: Buffer<i64> = (0..32_768)
        .flat_map(|_| {
            let v = rng.random_range(0..1i64 << 31);
            [v, v]
        })
        .collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

#[test]
fn test_plan_rejects_nesting_deeper_than_the_cascade() -> VortexResult<()> {
    let schemes: [&'static dyn Scheme; 1] = [&RunEndScheme];
    let nested = |depth: usize| {
        let id = RunEndScheme.id();
        format!("{}{}", format!("{id}(").repeat(depth), ")".repeat(depth)).replace("()", "(_)")
    };
    Plan::parse(&nested(MAX_CASCADE), &schemes)?;
    assert!(Plan::parse(&nested(MAX_CASCADE + 1), &schemes).is_err());
    assert!(Plan::parse(&nested(100_000), &schemes).is_err());
    Ok(())
}

#[test]
fn run_model_size_only_picks_runend_for_wide_short_runs() -> VortexResult<()> {

    use crate::schemes::integer::RunModel;

    let input = wide_runs_of_two();
    let default = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let default = assert_roundtrip(&default, &input)?;
    assert!(!default.is::<RunEnd>(), "{}", default.display_tree());

    let modeled = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .with_run_model(RunModel::size_only())
        .build();
    let compressed = assert_roundtrip(&modeled, &input)?;
    assert!(compressed.is::<RunEnd>(), "{}", compressed.display_tree());
    assert!(compressed.nbytes() < default.nbytes() * 2 / 3);
    Ok(())
}

#[test]
fn run_model_decode_cost_avoids_runend_at_memory_bandwidth() -> VortexResult<()> {

    use crate::schemes::integer::RunModel;

    let input = wide_runs_of_two();
    let modeled = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .with_run_model(RunModel::for_bandwidth(20e9))
        .build();
    let compressed = assert_roundtrip(&modeled, &input)?;
    assert!(!compressed.is::<RunEnd>(), "{}", compressed.display_tree());
    assert!(
        compressed.nbytes() < input.nbytes() * 3 / 4,
        "{}",
        compressed.display_tree()
    );
    Ok(())
}

#[test]
fn run_model_reuses_schemes_for_equal_models() {
    use crate::schemes::integer::RunModel;
    use crate::schemes::integer::modeled_schemes;

    let a = modeled_schemes(RunModel::for_bandwidth(2e9));
    let b = modeled_schemes(RunModel::for_bandwidth(2e9));
    assert!(a.iter().zip(&b).all(|(x, y)| std::ptr::addr_eq(*x, *y)));
}
