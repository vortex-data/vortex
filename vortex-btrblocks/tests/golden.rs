// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Golden-corpus determinism tests for the default compressor.
//!
//! Compresses a fixed, seed-generated corpus and snapshots each entry's full encoding tree
//! and exact byte counts. These snapshots pin the default compressor's *decisions*: any
//! refactor of scheme selection (see the compressor cost-model track) must leave every
//! snapshot untouched, so snapshot churn in a later change is the reviewable signal of a
//! behavior change.
//!
//! Three variants cover the feature matrix:
//!
//! - `default`: the default feature set and [`BtrBlocksCompressor::default`].
//! - `unstable`: `unstable_encodings` enabled, default builder — pins Delta / OnPair
//!   selection (compiled out of `ALL_SCHEMES` otherwise).
//! - `compact`: `unstable_encodings` + `zstd` + `pco`, with
//!   [`BtrBlocksCompressorBuilder::with_compact`] — pins Zstd / Pco selection.
//!
//! Every corpus entry is longer than 1024 values so the sampling-based estimation path is
//! exercised, and each entry is compressed twice per run to assert determinism directly.

#![allow(clippy::cast_possible_truncation, clippy::tests_outside_test_module)]

use std::fmt;
use std::sync::Arc;
use std::sync::LazyLock;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::display::EncodingSummaryExtractor;
use vortex_array::display::MetadataExtractor;
use vortex_array::display::TreeContext;
use vortex_array::display::TreeExtractor;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

/// Number of values in each numeric corpus entry: comfortably above the 1024-value sampling
/// threshold so scheme selection runs on sampled estimates, as it does for real file chunks.
const N: usize = 16_384;

/// Header extractor printing exact byte counts (the built-in [`NbytesExtractor`] rounds
/// through `humansize`, which could mask small size regressions).
///
/// [`NbytesExtractor`]: vortex_array::display::NbytesExtractor
struct ExactNbytesExtractor;

impl TreeExtractor<ArrayRef, TreeContext> for ExactNbytesExtractor {
    fn write_header(
        &self,
        array: &ArrayRef,
        _ctx: &TreeContext,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, " nbytes={}", array.nbytes())
    }
}

/// Renders the full snapshot content for one compressed corpus entry.
fn render(input: &ArrayRef, compressed: &ArrayRef) -> String {
    format!(
        "input: {}, len={}, nbytes={}\n{}",
        input.dtype(),
        input.len(),
        input.nbytes(),
        compressed
            .tree_display_builder()
            .with(EncodingSummaryExtractor)
            .with(ExactNbytesExtractor)
            .with(MetadataExtractor)
    )
}

/// Compresses every corpus entry twice (direct determinism check) and snapshots the result
/// under `{variant}__{entry}`.
fn golden_corpus_snapshots(variant: &str, compressor: &BtrBlocksCompressor) -> VortexResult<()> {
    for (name, array) in corpus()? {
        let rendered = {
            let mut exec_ctx = SESSION.create_execution_ctx();
            render(&array, &compressor.compress(&array, &mut exec_ctx)?)
        };
        let rendered_again = {
            let mut exec_ctx = SESSION.create_execution_ctx();
            render(&array, &compressor.compress(&array, &mut exec_ctx)?)
        };
        assert_eq!(
            rendered, rendered_again,
            "compressing corpus entry {name} twice produced different results"
        );

        insta::assert_snapshot!(format!("{variant}__{name}"), rendered);
    }
    Ok(())
}

/// The fixed corpus: deterministic synthetic arrays covering each scheme's habitat.
fn corpus() -> VortexResult<Vec<(&'static str, ArrayRef)>> {
    Ok(vec![
        ("int_monotone_jitter", int_monotone_jitter()),
        ("int_arithmetic_sequence", int_arithmetic_sequence()),
        ("int_low_cardinality", int_low_cardinality()),
        ("int_runs", int_runs()),
        ("int_sparse_outliers", int_sparse_outliers()),
        ("int_mostly_null", int_mostly_null()),
        ("int_negatives", int_negatives()),
        ("int_wide_random", int_wide_random()),
        ("float_alp_prices", float_alp_prices()),
        ("float_low_cardinality", float_low_cardinality()),
        ("float_full_precision", float_full_precision()),
        ("float_mostly_null", float_mostly_null()),
        ("string_fsst_structured", string_fsst_structured()),
        ("string_low_cardinality", string_low_cardinality()),
        ("binary_low_cardinality", binary_low_cardinality()),
        ("decimal_prices", decimal_prices()),
        ("temporal_timestamp_micros", temporal_timestamp_micros()),
        ("bool_random", bool_random()),
        ("struct_mixed", struct_mixed()?),
        ("list_of_int_runs", list_of_int_runs()?),
    ])
}

/// Near-monotone u64 (timestamp-like): FoR/BitPacking habitat, Delta habitat when enabled.
fn int_monotone_jitter() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(101);
    let mut value = 1_700_000_000_000u64;
    let values: Buffer<u64> = (0..N)
        .map(|_| {
            value += 900 + rng.random_range(0..200);
            value
        })
        .collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

/// Exact arithmetic sequence: Sequence habitat (distinct == len, no nulls).
fn int_arithmetic_sequence() -> ArrayRef {
    let values: Buffer<i64> = (0..N as i64).map(|i| 10_000 + 7 * i).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

/// A handful of widely-spaced distinct values: IntDict habitat.
fn int_low_cardinality() -> ArrayRef {
    const DISTINCT: [i64; 6] = [0, 123_400, 617_000, 1_234_000, 12_340_000, 37_020_000];
    let mut rng = StdRng::seed_from_u64(102);
    let values: Buffer<i64> = (0..N).map(|_| DISTINCT[rng.random_range(0..6)]).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

/// Long runs over a moderate value set: RunEnd/RLE habitat.
fn int_runs() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(103);
    let mut values: Vec<i32> = Vec::with_capacity(N);
    while values.len() < N {
        let value = rng.random_range(-50_000..50_000i32);
        let run = rng.random_range(8..25);
        values.extend(std::iter::repeat_n(value, run));
    }
    values.truncate(N);
    PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable).into_array()
}

/// One dominant value with rare large outliers: Sparse habitat.
fn int_sparse_outliers() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(104);
    let values: Buffer<i64> = (0..N)
        .map(|_| {
            if rng.random_range(0..100) < 5 {
                rng.random_range(1_000_000_000..2_000_000_000i64)
            } else {
                1_000_000
            }
        })
        .collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

/// 95% nulls over small values: null-dominated integer habitat.
fn int_mostly_null() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(105);
    let mut validity: Vec<bool> = Vec::with_capacity(N);
    let values: Buffer<i32> = (0..N)
        .map(|_| {
            let valid = rng.random_range(0..100) < 5;
            validity.push(valid);
            if valid { rng.random_range(0..1000) } else { 0 }
        })
        .collect();
    PrimitiveArray::new(
        values,
        Validity::Array(BoolArray::from_iter(validity).into_array()),
    )
    .into_array()
}

/// Small values of mixed sign: ZigZag habitat.
fn int_negatives() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(106);
    let values: Buffer<i64> = (0..N).map(|_| rng.random_range(-128..128i64)).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

/// Full-width random u64: essentially incompressible; pins the "no scheme wins" path.
fn int_wide_random() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(107);
    let values: Buffer<u64> = (0..N).map(|_| rng.random::<u64>()).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

/// Two-decimal-digit "prices": ALP habitat.
fn float_alp_prices() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(108);
    let values: Buffer<f64> = (0..N)
        .map(|_| rng.random_range(0..10_000_000i64) as f64 / 100.0)
        .collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

/// A handful of distinct floats: FloatDict habitat.
fn float_low_cardinality() -> ArrayRef {
    const DISTINCT: [f64; 8] = [0.0, 0.5, 1.25, 2.75, 3.5, 10.125, 100.0625, 1000.03125];
    let mut rng = StdRng::seed_from_u64(109);
    let values: Buffer<f64> = (0..N).map(|_| DISTINCT[rng.random_range(0..8)]).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

/// Full-precision uniform floats: ALP-RD habitat.
fn float_full_precision() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(110);
    let values: Buffer<f64> = (0..N).map(|_| rng.random::<f64>()).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

/// 95% nulls over floats: null-dominated sparse float habitat.
fn float_mostly_null() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(111);
    let mut validity: Vec<bool> = Vec::with_capacity(N);
    let values: Buffer<f64> = (0..N)
        .map(|_| {
            let valid = rng.random_range(0..100) < 5;
            validity.push(valid);
            if valid {
                rng.random::<f64>() * 100.0
            } else {
                0.0
            }
        })
        .collect();
    PrimitiveArray::new(
        values,
        Validity::Array(BoolArray::from_iter(validity).into_array()),
    )
    .into_array()
}

/// High-cardinality strings with shared substructure (emails): FSST habitat.
fn string_fsst_structured() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(112);
    let strings: Vec<String> = (0..N)
        .map(|_| {
            format!(
                "user{:06}@example{}.com",
                rng.random_range(0..1_000_000),
                rng.random_range(0..100)
            )
        })
        .collect();
    VarBinViewArray::from_iter_str(strings.iter().map(String::as_str)).into_array()
}

/// A dozen distinct strings: StringDict habitat.
fn string_low_cardinality() -> ArrayRef {
    const DISTINCT: [&str; 12] = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliett", "kilo", "lima",
    ];
    let mut rng = StdRng::seed_from_u64(113);
    let strings: Vec<Option<&str>> = (0..N)
        .map(|_| Some(DISTINCT[rng.random_range(0..12)]))
        .collect();
    VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable)).into_array()
}

/// Low-cardinality binary blobs: BinaryDict habitat (Zstd habitat under `compact`).
fn binary_low_cardinality() -> ArrayRef {
    const DISTINCT: [&[u8]; 5] = [
        &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
        &[0xDE, 0xAD, 0xBE, 0xEF],
        &[0x00; 16],
        &[0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00],
        &[0x42; 12],
    ];
    let mut rng = StdRng::seed_from_u64(114);
    let blobs: Vec<Option<&[u8]>> = (0..N)
        .map(|_| Some(DISTINCT[rng.random_range(0..5)]))
        .collect();
    VarBinViewArray::from_iter(blobs, DType::Binary(Nullability::NonNullable)).into_array()
}

/// Two-decimal-place decimals: DecimalScheme (byte-parts) habitat.
fn decimal_prices() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(115);
    let values: Buffer<i64> = (0..N).map(|_| rng.random_range(0..10_000_000i64)).collect();
    DecimalArray::new(values, DecimalDType::new(12, 2), Validity::NonNullable).into_array()
}

/// Near-monotone microsecond timestamps: TemporalScheme (datetime-parts) habitat.
fn temporal_timestamp_micros() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(116);
    let mut value = 1_700_000_000_000_000i64;
    let values: Buffer<i64> = (0..N)
        .map(|_| {
            value += rng.random_range(1_000..1_000_000);
            value
        })
        .collect();
    TemporalArray::new_timestamp(
        PrimitiveArray::new(values, Validity::NonNullable).into_array(),
        TimeUnit::Microseconds,
        Some(Arc::from("UTC")),
    )
    .into_array()
}

/// Random booleans: no bool scheme is registered, pinning the "stays canonical" path.
fn bool_random() -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(117);
    BoolArray::from_iter((0..N).map(|_| rng.random::<bool>())).into_array()
}

/// Struct of int/string/float fields: pins the structural recursion path.
fn struct_mixed() -> VortexResult<ArrayRef> {
    Ok(StructArray::from_fields(&[
        ("id", int_arithmetic_sequence()),
        ("category", string_low_cardinality()),
        ("value", float_alp_prices()),
    ])?
    .into_array())
}

/// Variable-length lists of run-heavy ints: pins the list offsets/elements path.
fn list_of_int_runs() -> VortexResult<ArrayRef> {
    let mut rng = StdRng::seed_from_u64(118);
    let elements = int_runs();
    let mut offsets: Vec<i32> = Vec::with_capacity(N / 4 + 1);
    let mut offset = 0i32;
    offsets.push(offset);
    while (offset as usize) < N {
        offset = (offset + rng.random_range(1..8)).min(N as i32);
        offsets.push(offset);
    }
    let offsets = PrimitiveArray::new(Buffer::copy_from(&offsets), Validity::NonNullable);
    Ok(ListArray::try_new(elements, offsets.into_array(), Validity::NonNullable)?.into_array())
}

/// Excludes OnPair from the golden compressors: its dictionary training (upstream `onpair`
/// crate) iterates randomly-seeded `hashbrown` maps, so its compressed output — and therefore
/// its sampled estimate — differs run-to-run. A nondeterministic scheme cannot serve as a
/// golden baseline; excluding it keeps the remaining unstable schemes pinned.
#[cfg(feature = "unstable_encodings")]
fn without_onpair(
    builder: vortex_btrblocks::BtrBlocksCompressorBuilder,
) -> vortex_btrblocks::BtrBlocksCompressorBuilder {
    use vortex_btrblocks::SchemeExt;
    use vortex_btrblocks::schemes::string::OnPairScheme;

    builder.exclude_schemes([OnPairScheme.id()])
}

#[cfg(not(feature = "unstable_encodings"))]
#[test]
fn golden_default() -> VortexResult<()> {
    golden_corpus_snapshots("default", &BtrBlocksCompressor::default())
}

#[cfg(feature = "unstable_encodings")]
#[test]
fn golden_unstable() -> VortexResult<()> {
    use vortex_btrblocks::BtrBlocksCompressorBuilder;

    golden_corpus_snapshots(
        "unstable",
        &without_onpair(BtrBlocksCompressorBuilder::default()).build(),
    )
}

#[cfg(all(feature = "unstable_encodings", feature = "zstd", feature = "pco"))]
#[test]
fn golden_compact() -> VortexResult<()> {
    use vortex_btrblocks::BtrBlocksCompressorBuilder;

    golden_corpus_snapshots(
        "compact",
        &without_onpair(BtrBlocksCompressorBuilder::default().with_compact()).build(),
    )
}
