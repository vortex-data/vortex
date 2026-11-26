// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
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
use crate::scheme::SchemeExt;
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

#[test]
fn constant_chunk_preserves_cached_plan_baseline() -> VortexResult<()> {
    let _lock = LOCK.lock();
    let compressor = compressor();
    let reuse = PlanReuse::default();

    let (_, estimates) = compress_counting(&compressor, &low_cardinality(0), &reuse)?;
    assert!(estimates > 0);

    // A constant chunk short-circuits before scheme selection and must not trigger re-selection
    // or overwrite the cached baseline ratio.
    let constant =
        PrimitiveArray::new(Buffer::<i64>::full(5, 4096), Validity::NonNullable).into_array();
    let (compressed, estimates) = compress_counting(&compressor, &constant, &reuse)?;
    assert_eq!(estimates, 0);
    assert_arrays_eq!(compressed, constant, &mut SESSION.create_execution_ctx());

    // The next Dict-friendly chunk still replays against the original baseline with zero estimates.
    let next_input = low_cardinality(100);
    let (next, estimates) = compress_counting(&compressor, &next_input, &reuse)?;
    assert!(next.is::<Dict>());
    assert_eq!(estimates, 0);
    assert_arrays_eq!(next, next_input, &mut SESSION.create_execution_ctx());

    Ok(())
}

#[test]
fn unchanged_chunk_preserves_cached_plan_baseline() -> VortexResult<()> {
    let _lock = LOCK.lock();
    let compressor = compressor();
    let reuse = PlanReuse::default();

    let (_, estimates) = compress_counting(&compressor, &low_cardinality(0), &reuse)?;
    assert!(estimates > 0);

    // An incompressible chunk still runs fallback selection once (without repeating full selection
    // a second time), and leaving it unchanged must not overwrite the cached plan's baseline ratio
    // with 1.0.
    let unique_input = unique();
    let (unchanged, estimates) = compress_counting(&compressor, &unique_input, &reuse)?;
    assert_eq!(estimates, 1);
    assert!(!unchanged.is::<Dict>());
    assert_arrays_eq!(unchanged, unique_input, &mut SESSION.create_execution_ctx());

    let next_input = low_cardinality(100);
    let (next, estimates) = compress_counting(&compressor, &next_input, &reuse)?;
    assert!(next.is::<Dict>());
    assert_eq!(estimates, 0);
    assert_arrays_eq!(next, next_input, &mut SESSION.create_execution_ctx());

    Ok(())
}

static DETACHED_ROOT_A_ELIGIBLE: AtomicBool = AtomicBool::new(true);
static DETACHED_ROOT_B_ELIGIBLE: AtomicBool = AtomicBool::new(true);
static DETACHED_ROOT_A_CALLS: AtomicUsize = AtomicUsize::new(0);
static DETACHED_ROOT_B_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct DetachedRootAScheme;

impl Scheme for DetachedRootAScheme {
    fn scheme_name(&self) -> &'static str {
        "test.detached_root_a"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        DETACHED_ROOT_A_ELIGIBLE.load(Ordering::SeqCst) && IntDictScheme.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.cascade_history().is_empty() {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(12.0))
        } else {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        DETACHED_ROOT_A_CALLS.fetch_add(1, Ordering::SeqCst);
        dict_with_padding(data.array(), 8)
    }
}

#[derive(Debug)]
struct DetachedRootBScheme;

impl Scheme for DetachedRootBScheme {
    fn scheme_name(&self) -> &'static str {
        "test.detached_root_b"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        DETACHED_ROOT_B_ELIGIBLE.load(Ordering::SeqCst) && IntDictScheme.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.cascade_history().is_empty() {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(10.0))
        } else {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        DETACHED_ROOT_B_CALLS.fetch_add(1, Ordering::SeqCst);
        dict_with_padding(data.array(), 8)
    }
}

#[test]
fn detached_chunks_commit_in_order() -> VortexResult<()> {
    let _lock = LOCK.lock();
    DETACHED_ROOT_A_ELIGIBLE.store(false, Ordering::SeqCst);
    DETACHED_ROOT_B_ELIGIBLE.store(false, Ordering::SeqCst);
    DETACHED_ROOT_A_CALLS.store(0, Ordering::SeqCst);
    DETACHED_ROOT_B_CALLS.store(0, Ordering::SeqCst);

    let compressor = CascadingCompressor::new(vec![
        &IntDictScheme,
        &DetachedRootAScheme,
        &DetachedRootBScheme,
        &CompetitorScheme,
    ]);
    let reuse = PlanReuse::new(PlanReuseOptions {
        challenge_after: None,
        ..PlanReuseOptions::default()
    });
    let input = low_cardinality(0);

    // Chunk 0 establishes IntDictScheme as the initial cached plan.
    let chunk0 = reuse.next_chunk();
    assert!(!reuse.has_plan());
    let (first, estimates) = compress_counting(&compressor, &input, &chunk0)?;
    assert!(first.is::<Dict>());
    assert!(estimates > 0);
    reuse.commit(chunk0);
    assert!(reuse.has_plan());

    // Reserve chunk1 and chunk2 detached from the same initial plan, compress them out of order
    // so chunk1 records Replace(DetachedRootA) and chunk2 records Replace(DetachedRootB), then
    // commit in chunk order (chunk1, then chunk2).
    let chunk1 = reuse.next_chunk();
    let chunk2 = reuse.next_chunk();

    // Compress chunk2 first: IntDictScheme produces ~4160 bytes, while DetachedRootB also matches
    // when IntDictScheme is forced to deviate via a different-size compressor — use a separate
    // compressor without IntDictScheme so each detached chunk falls back to its own scheme.
    let compressor_a = CascadingCompressor::new(vec![&DetachedRootAScheme, &CompetitorScheme]);
    let compressor_b = CascadingCompressor::new(vec![&DetachedRootBScheme, &CompetitorScheme]);

    DETACHED_ROOT_B_ELIGIBLE.store(true, Ordering::SeqCst);
    let (_, estimates2) = compress_counting(&compressor_b, &input, &chunk2)?;
    assert!(estimates2 > 0);

    DETACHED_ROOT_A_ELIGIBLE.store(true, Ordering::SeqCst);
    let (_, estimates1) = compress_counting(&compressor_a, &input, &chunk1)?;
    assert!(estimates1 > 0);

    reuse.commit(chunk1);
    reuse.commit(chunk2);

    // Chunk 3 must replay the last committed plan (DetachedRootB from chunk2) with zero estimates;
    // swapping the commit order would replay DetachedRootA instead and fail this assertion.
    DETACHED_ROOT_A_CALLS.store(0, Ordering::SeqCst);
    DETACHED_ROOT_B_CALLS.store(0, Ordering::SeqCst);
    let chunk3 = reuse.next_chunk();
    let (third, estimates3) = compress_counting(&compressor, &input, &chunk3)?;
    assert_eq!(estimates3, 0);
    assert_eq!(DETACHED_ROOT_A_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(DETACHED_ROOT_B_CALLS.load(Ordering::SeqCst), 1);
    assert_arrays_eq!(third, input, &mut SESSION.create_execution_ctx());

    Ok(())
}

#[test]
fn detached_waves_preserve_challenge_cadence() -> VortexResult<()> {
    let _lock = LOCK.lock();
    let compressor = compressor();
    let reuse = PlanReuse::new(PlanReuseOptions {
        challenge_after: Some(8),
        ..PlanReuseOptions::default()
    });

    // Establish the initial plan from chunk 0 (as the layout writer does before wave batching).
    let chunk0 = reuse.next_chunk();
    let (_, estimates0) = compress_counting(&compressor, &low_cardinality(0), &chunk0)?;
    assert!(estimates0 > 0);
    reuse.commit(chunk0);

    // Process chunks 1..40 in 16-chunk detached waves and verify challenges fire every 9 chunks
    // (chunks 9, 18, 27, 36), matching `challenge_after: Some(8)`.
    let mut challenged = Vec::new();
    let mut idx = 1usize;
    while idx < 40 {
        let wave_end = (idx + 16).min(40);
        let wave: Vec<_> = (idx..wave_end).map(|i| (i, reuse.next_chunk())).collect();
        for (i, chunk) in wave {
            let (_, estimates) =
                compress_counting(&compressor, &low_cardinality(i as i64), &chunk)?;
            if estimates > 0 {
                challenged.push(i);
            }
            reuse.commit(chunk);
        }
        idx = wave_end;
    }

    assert_eq!(challenged, vec![9, 18, 27, 36]);

    Ok(())
}

/// Encodes `i64` values in `0..8` as a `DictArray` with `u8` codes and `values_len` `i64` values,
/// so the output always round-trips identically while its size is controlled by `values_len`.
fn dict_with_padding(array: &ArrayRef, values_len: usize) -> VortexResult<ArrayRef> {
    let prim = array.as_opt::<vortex_array::arrays::Primitive>().unwrap();
    let codes: Buffer<u8> = prim
        .as_slice::<i64>()
        .iter()
        .map(|&v| u8::try_from(v).unwrap())
        .collect();
    let values: Buffer<i64> = (0..values_len as i64).collect();
    Ok(DictArray::try_new(
        PrimitiveArray::new(codes, Validity::NonNullable).into_array(),
        PrimitiveArray::new(values, Validity::NonNullable).into_array(),
    )?
    .into_array())
}

/// Root-only cascading scheme that builds a `DictArray` with `u8` codes and cascades `values`
/// (64 `i64`s = 512 bytes) to child index 0. Because `codes` is 4,096 bytes, large relative ratio
/// changes on `values` keep the root compression ratio stable.
#[derive(Debug)]
struct CascadeRootScheme;

impl Scheme for CascadeRootScheme {
    fn scheme_name(&self) -> &'static str {
        "test.cascade_root"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        IntDictScheme.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.cascade_history().is_empty() {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(10.0))
        } else {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let prim = data
            .array()
            .as_opt::<vortex_array::arrays::Primitive>()
            .unwrap();
        let codes: Buffer<u8> = prim
            .as_slice::<i64>()
            .iter()
            .map(|&v| u8::try_from(v).unwrap())
            .collect();
        // 64 values in `0..8` repeated so child schemes can compress `values` via `dict_with_padding`
        // while preserving exact lookup for all code indices `0..8`.
        let values: Buffer<i64> = (0..64i64).map(|i| i % 8).collect();
        let values_array = PrimitiveArray::new(values, Validity::NonNullable).into_array();
        let compressed_values =
            compressor.compress_child(&values_array, &compress_ctx, self.id(), 0, exec_ctx)?;
        Ok(DictArray::try_new(
            PrimitiveArray::new(codes, Validity::NonNullable).into_array(),
            compressed_values,
        )?
        .into_array())
    }
}

static CHILD_A_ELIGIBLE: AtomicBool = AtomicBool::new(true);
static CHILD_A_PADDING: AtomicUsize = AtomicUsize::new(8);
static CHILD_B_PADDING: AtomicUsize = AtomicUsize::new(8);
static CHILD_B_RATIO: AtomicUsize = AtomicUsize::new(4);
static CHILD_B_CALLS: AtomicUsize = AtomicUsize::new(0);
static GRANDCHILD_REPLAY_CALLS: AtomicUsize = AtomicUsize::new(0);
static GRANDCHILD_FALLBACK_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Depth-1 child scheme that compresses `values` and cascades a child to depth 2.
#[derive(Debug)]
struct ChildAScheme;

impl Scheme for ChildAScheme {
    fn scheme_name(&self) -> &'static str {
        "test.child_a"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        CHILD_A_ELIGIBLE.load(Ordering::SeqCst) && IntDictScheme.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.cascade_history().len() == 1 {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(8.0))
        } else {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let compressed =
            compressor.compress_child(data.array(), &compress_ctx, self.id(), 0, exec_ctx)?;
        if compressed.is::<Dict>() {
            Ok(compressed)
        } else {
            dict_with_padding(data.array(), CHILD_A_PADDING.load(Ordering::SeqCst))
        }
    }
}

/// Alternative depth-1 child scheme.
#[derive(Debug)]
struct ChildBScheme;

impl Scheme for ChildBScheme {
    fn scheme_name(&self) -> &'static str {
        "test.child_b"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        IntDictScheme.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.cascade_history().len() == 1 {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(
                CHILD_B_RATIO.load(Ordering::SeqCst) as f64,
            ))
        } else {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        CHILD_B_CALLS.fetch_add(1, Ordering::SeqCst);
        let compressed =
            compressor.compress_child(data.array(), &compress_ctx, self.id(), 0, exec_ctx)?;
        if compressed.is::<Dict>() {
            Ok(compressed)
        } else {
            dict_with_padding(data.array(), CHILD_B_PADDING.load(Ordering::SeqCst))
        }
    }
}

/// Depth-2 grandchild scheme under `ChildAScheme`.
#[derive(Debug)]
struct GrandchildAScheme;

impl Scheme for GrandchildAScheme {
    fn scheme_name(&self) -> &'static str {
        "test.grandchild_a"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        IntDictScheme.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.cascade_history().len() == 2
            && compress_ctx.cascade_history()[1].0 == ChildAScheme.id()
        {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(6.0))
        } else {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        GRANDCHILD_REPLAY_CALLS.fetch_add(1, Ordering::SeqCst);
        dict_with_padding(data.array(), CHILD_A_PADDING.load(Ordering::SeqCst))
    }
}

/// Depth-2 grandchild scheme under `ChildBScheme`.
#[derive(Debug)]
struct GrandchildBScheme;

impl Scheme for GrandchildBScheme {
    fn scheme_name(&self) -> &'static str {
        "test.grandchild_b"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        IntDictScheme.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        if compress_ctx.cascade_history().len() == 2
            && compress_ctx.cascade_history()[1].0 == ChildBScheme.id()
        {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(6.0))
        } else {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        GRANDCHILD_FALLBACK_CALLS.fetch_add(1, Ordering::SeqCst);
        dict_with_padding(data.array(), CHILD_B_PADDING.load(Ordering::SeqCst))
    }
}

#[test]
fn child_replay_unchanged_skips_estimates() -> VortexResult<()> {
    let _lock = LOCK.lock();
    let compressor = CascadingCompressor::new(vec![&CascadeRootScheme, &CompetitorScheme]);
    let reuse = PlanReuse::default();

    // Chunk 0 selects CascadeRootScheme at root and records Decision::Unchanged at child 0.
    let input0 = low_cardinality(0);
    let (first, estimates) = compress_counting(&compressor, &input0, &reuse)?;
    assert!(estimates > 0);
    assert_arrays_eq!(first, input0, &mut SESSION.create_execution_ctx());

    // Chunk 1 replays CascadeRootScheme and replays Decision::Unchanged at child 0 without
    // estimating CompetitorScheme.
    let input1 = low_cardinality(0);
    let (second, estimates) = compress_counting(&compressor, &input1, &reuse)?;
    assert_eq!(estimates, 0);
    assert_arrays_eq!(second, input1, &mut SESSION.create_execution_ctx());

    Ok(())
}

#[test]
fn child_replay_ineligible_falls_back_and_records_replacement() -> VortexResult<()> {
    let _lock = LOCK.lock();
    CHILD_A_ELIGIBLE.store(true, Ordering::SeqCst);
    CHILD_A_PADDING.store(8, Ordering::SeqCst);
    CHILD_B_PADDING.store(16, Ordering::SeqCst);
    CHILD_B_RATIO.store(4, Ordering::SeqCst);
    CHILD_B_CALLS.store(0, Ordering::SeqCst);

    let compressor =
        CascadingCompressor::new(vec![&CascadeRootScheme, &ChildAScheme, &ChildBScheme]);
    let reuse = PlanReuse::default();
    let input = low_cardinality(0);
    let mut ctx = SESSION.create_execution_ctx();

    // Chunk 0 plans CascadeRootScheme -> ChildAScheme.
    let (first, recorded0) = compressor.compress_with_plan(&input, None, &reuse, &mut ctx)?;
    assert_arrays_eq!(first, input, &mut ctx);
    assert_eq!(CHILD_B_CALLS.load(Ordering::SeqCst), 0);

    // Make ChildAScheme ineligible on chunk 1: child replay falls back to ChildBScheme and
    // records ChildBScheme in the plan.
    CHILD_A_ELIGIBLE.store(false, Ordering::SeqCst);
    let (second, recorded1) = compressor.compress_with_plan(
        &input,
        Some(std::sync::Arc::new(recorded0.plan)),
        &reuse,
        &mut ctx,
    )?;
    assert!(recorded1.child_replanned);
    assert_arrays_eq!(second, input, &mut ctx);
    assert_eq!(CHILD_B_CALLS.load(Ordering::SeqCst), 1);

    // Replaying `recorded1.plan` with ChildAScheme eligible again replays ChildBScheme directly.
    CHILD_A_ELIGIBLE.store(true, Ordering::SeqCst);
    let (third, ..) = compressor.compress_with_plan(
        &input,
        Some(std::sync::Arc::new(recorded1.plan)),
        &reuse,
        &mut ctx,
    )?;
    assert_arrays_eq!(third, input, &mut ctx);
    assert_eq!(CHILD_B_CALLS.load(Ordering::SeqCst), 2);

    Ok(())
}

#[test]
fn child_replan_persists_across_compress_reusing() -> VortexResult<()> {
    let _lock = LOCK.lock();
    CHILD_A_ELIGIBLE.store(true, Ordering::SeqCst);
    CHILD_A_PADDING.store(8, Ordering::SeqCst);
    CHILD_B_PADDING.store(8, Ordering::SeqCst);
    CHILD_B_RATIO.store(4, Ordering::SeqCst);
    CHILD_B_CALLS.store(0, Ordering::SeqCst);

    let compressor = CascadingCompressor::new(vec![
        &CascadeRootScheme,
        &ChildAScheme,
        &ChildBScheme,
        &CompetitorScheme,
    ]);
    let reuse = PlanReuse::new(PlanReuseOptions {
        max_ratio_deviation: 0.1,
        challenge_after: None,
    });
    let input = low_cardinality(0);

    // Chunk 0 plans CascadeRootScheme -> ChildAScheme.
    let (first, estimates0) = compress_counting(&compressor, &input, &reuse)?;
    assert!(estimates0 > 0);
    assert_arrays_eq!(first, input, &mut SESSION.create_execution_ctx());
    assert_eq!(CHILD_B_CALLS.load(Ordering::SeqCst), 0);

    // Chunk 1: ChildAScheme deviates (>10% child ratio drop) while root ratio stays within 10%.
    // Child fallback picks ChildBScheme (smaller), and `compress_reusing` persists the child replan.
    CHILD_A_PADDING.store(48, Ordering::SeqCst);
    CHILD_B_PADDING.store(8, Ordering::SeqCst);
    CHILD_B_RATIO.store(12, Ordering::SeqCst);
    let (second, estimates1) = compress_counting(&compressor, &input, &reuse)?;
    assert!(estimates1 > 0);
    assert_arrays_eq!(second, input, &mut SESSION.create_execution_ctx());
    assert_eq!(CHILD_B_CALLS.load(Ordering::SeqCst), 1);

    // Chunk 2: replays ChildBScheme directly with zero competitor estimates.
    let (third, estimates2) = compress_counting(&compressor, &input, &reuse)?;
    assert_eq!(estimates2, 0);
    assert_arrays_eq!(third, input, &mut SESSION.create_execution_ctx());
    assert_eq!(CHILD_B_CALLS.load(Ordering::SeqCst), 2);

    Ok(())
}

#[test]
fn child_deviation_keeps_smaller_and_preserves_winning_subtree() -> VortexResult<()> {
    let _lock = LOCK.lock();
    CHILD_A_ELIGIBLE.store(true, Ordering::SeqCst);
    CHILD_A_PADDING.store(8, Ordering::SeqCst);
    CHILD_B_PADDING.store(40, Ordering::SeqCst);
    CHILD_B_RATIO.store(4, Ordering::SeqCst);
    CHILD_B_CALLS.store(0, Ordering::SeqCst);
    GRANDCHILD_REPLAY_CALLS.store(0, Ordering::SeqCst);
    GRANDCHILD_FALLBACK_CALLS.store(0, Ordering::SeqCst);

    let compressor = CascadingCompressor::new(vec![
        &CascadeRootScheme,
        &ChildAScheme,
        &ChildBScheme,
        &GrandchildAScheme,
        &GrandchildBScheme,
    ]);
    let reuse = PlanReuse::new(PlanReuseOptions {
        max_ratio_deviation: 0.1,
        challenge_after: None,
    });
    let input = low_cardinality(0);
    let mut ctx = SESSION.create_execution_ctx();

    // Chunk 0 establishes: CascadeRootScheme -> ChildAScheme -> GrandchildAScheme.
    let (first, recorded0) = compressor.compress_with_plan(&input, None, &reuse, &mut ctx)?;
    assert_arrays_eq!(first, input, &mut ctx);
    assert_eq!(GRANDCHILD_REPLAY_CALLS.load(Ordering::SeqCst), 1);

    // Chunk 1: ChildAScheme deviates (values padding 8 -> 24). Fallback prefers ChildBScheme by
    // estimate, which cascades to GrandchildBScheme (padding 40 => larger), so the replayed
    // ChildAScheme wins and must restore its subtree without leaking GrandchildBScheme.
    CHILD_A_PADDING.store(24, Ordering::SeqCst);
    CHILD_B_PADDING.store(40, Ordering::SeqCst);
    CHILD_B_RATIO.store(12, Ordering::SeqCst);
    GRANDCHILD_REPLAY_CALLS.store(0, Ordering::SeqCst);
    GRANDCHILD_FALLBACK_CALLS.store(0, Ordering::SeqCst);

    let (second, recorded1) = compressor.compress_with_plan(
        &input,
        Some(std::sync::Arc::new(recorded0.plan)),
        &reuse,
        &mut ctx,
    )?;
    assert_arrays_eq!(second, input, &mut ctx);
    assert!(GRANDCHILD_REPLAY_CALLS.load(Ordering::SeqCst) >= 1);
    assert_eq!(GRANDCHILD_FALLBACK_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(
        recorded1
            .plan
            .decision(&[(CascadeRootScheme.id(), 0), (ChildBScheme.id(), 0)]),
        None,
        "losing fallback subtree must not remain in recorded plan"
    );
    let second_bytes = second.nbytes();

    // Chunk 2: replaying `recorded1.plan` uses ChildAScheme -> GrandchildAScheme directly.
    GRANDCHILD_REPLAY_CALLS.store(0, Ordering::SeqCst);
    GRANDCHILD_FALLBACK_CALLS.store(0, Ordering::SeqCst);
    let plan1 = std::sync::Arc::new(recorded1.plan);
    let (third, ..) = compressor.compress_with_plan(
        &input,
        Some(std::sync::Arc::clone(&plan1)),
        &reuse,
        &mut ctx,
    )?;
    assert_arrays_eq!(third, input, &mut ctx);
    assert_eq!(third.nbytes(), second_bytes);
    assert_eq!(GRANDCHILD_REPLAY_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(GRANDCHILD_FALLBACK_CALLS.load(Ordering::SeqCst), 0);

    // Chunk 3: trigger child deviation again with ChildBScheme smaller than replayed ChildAScheme;
    // fallback must win, drop ChildAScheme's stale subtree, and chunk 4 must replay ChildBScheme.
    CHILD_A_PADDING.store(48, Ordering::SeqCst);
    CHILD_B_PADDING.store(8, Ordering::SeqCst);
    GRANDCHILD_REPLAY_CALLS.store(0, Ordering::SeqCst);
    GRANDCHILD_FALLBACK_CALLS.store(0, Ordering::SeqCst);

    let (fourth, recorded3) =
        compressor.compress_with_plan(&input, Some(plan1), &reuse, &mut ctx)?;
    assert_arrays_eq!(fourth, input, &mut ctx);
    assert!(fourth.nbytes() < second_bytes);
    assert_eq!(GRANDCHILD_FALLBACK_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(
        recorded3
            .plan
            .decision(&[(CascadeRootScheme.id(), 0), (ChildAScheme.id(), 0)]),
        None,
        "losing replayed subtree must not remain in recorded plan"
    );

    GRANDCHILD_REPLAY_CALLS.store(0, Ordering::SeqCst);
    GRANDCHILD_FALLBACK_CALLS.store(0, Ordering::SeqCst);
    let (fifth, ..) = compressor.compress_with_plan(
        &input,
        Some(std::sync::Arc::new(recorded3.plan)),
        &reuse,
        &mut ctx,
    )?;
    assert_arrays_eq!(fifth, input, &mut ctx);
    assert_eq!(fifth.nbytes(), fourth.nbytes());
    assert_eq!(GRANDCHILD_REPLAY_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(GRANDCHILD_FALLBACK_CALLS.load(Ordering::SeqCst), 1);

    Ok(())
}
