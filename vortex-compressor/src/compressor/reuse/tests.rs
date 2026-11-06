// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::PlanReuse;
use super::PlanReuseOptions;
use crate::CascadingCompressor;
use crate::builtins::IntDictScheme;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::stats::ArrayAndStats;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// Serializes the tests, because they share [`ESTIMATES`].
static LOCK: Mutex<()> = Mutex::new(());

/// The number of estimates that [`CompetitorScheme`] made. Only a full scheme selection asks it.
static ESTIMATES: AtomicUsize = AtomicUsize::new(0);

/// A scheme that counts its estimates and never wins.
#[derive(Debug)]
struct CompetitorScheme;

impl Scheme for CompetitorScheme {
    fn scheme_name(&self) -> &'static str {
        "test.competitor"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        IntDictScheme.matches(canonical)
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
        ESTIMATES.fetch_add(1, Ordering::SeqCst);
        CompressionEstimate::Verdict(EstimateVerdict::Skip)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unreachable!("the competitor never wins")
    }
}

fn compressor() -> CascadingCompressor {
    CascadingCompressor::new(vec![&IntDictScheme, &CompetitorScheme])
}

/// Returns 4,096 integers with 8 distinct values, offset by `seed`.
fn low_cardinality(seed: i64) -> ArrayRef {
    let values: Buffer<i64> = (0..4096i64).map(|i| seed + (i * 7) % 8).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

/// Returns 4,096 distinct integers.
fn unique() -> ArrayRef {
    let values: Buffer<i64> = (0..4096i64).map(|i| i * 1_000_003).collect();
    PrimitiveArray::new(values, Validity::NonNullable).into_array()
}

/// Compresses `array` through `reuse` and returns the result and the number of competitor
/// estimates.
fn compress_counting(
    compressor: &CascadingCompressor,
    array: &ArrayRef,
    reuse: &PlanReuse,
) -> VortexResult<(ArrayRef, usize)> {
    ESTIMATES.store(0, Ordering::SeqCst);
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress_reusing(array, reuse, &mut ctx)?;

    Ok((compressed, ESTIMATES.load(Ordering::SeqCst)))
}

#[test]
fn replays_plan_without_estimates() -> VortexResult<()> {
    let _lock = LOCK.lock();
    let compressor = compressor();
    let reuse = PlanReuse::default();

    let (first, estimates) = compress_counting(&compressor, &low_cardinality(0), &reuse)?;
    assert!(first.is::<Dict>());
    assert!(estimates > 0);

    let second_input = low_cardinality(100);
    let (second, estimates) = compress_counting(&compressor, &second_input, &reuse)?;
    assert!(second.is::<Dict>());
    assert_eq!(estimates, 0);
    assert_arrays_eq!(second, second_input, &mut SESSION.create_execution_ctx());

    Ok(())
}

#[test]
fn ratio_drop_selects_again() -> VortexResult<()> {
    let _lock = LOCK.lock();
    let compressor = compressor();
    let reuse = PlanReuse::default();

    compress_counting(&compressor, &low_cardinality(0), &reuse)?;

    // A dictionary of distinct values does not compress, so the replayed ratio drops and the
    // compressor selects again.
    let input = unique();
    let (compressed, estimates) = compress_counting(&compressor, &input, &reuse)?;
    assert!(estimates > 0);
    assert!(!compressed.is::<Dict>());
    assert_arrays_eq!(compressed, input, &mut SESSION.create_execution_ctx());

    Ok(())
}

#[test]
fn challenges_after_limit() -> VortexResult<()> {
    let _lock = LOCK.lock();
    let compressor = compressor();
    let reuse = PlanReuse::new(PlanReuseOptions {
        challenge_after: Some(1),
        ..PlanReuseOptions::default()
    });

    let (_, estimates) = compress_counting(&compressor, &low_cardinality(0), &reuse)?;
    assert!(estimates > 0);

    let (_, estimates) = compress_counting(&compressor, &low_cardinality(1), &reuse)?;
    assert_eq!(estimates, 0);

    let (_, estimates) = compress_counting(&compressor, &low_cardinality(2), &reuse)?;
    assert!(estimates > 0);

    Ok(())
}

#[test]
fn constant_chunk_makes_no_plan() -> VortexResult<()> {
    let _lock = LOCK.lock();
    let compressor = compressor();
    let reuse = PlanReuse::default();

    let constant = PrimitiveArray::new(Buffer::<i64>::full(5, 4096), Validity::NonNullable);
    let (_, estimates) = compress_counting(&compressor, &constant.into_array(), &reuse)?;
    assert_eq!(estimates, 0);

    let (_, estimates) = compress_counting(&compressor, &low_cardinality(0), &reuse)?;
    assert!(estimates > 0);

    Ok(())
}
