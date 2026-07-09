// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::CascadingCompressor;
use super::ROOT_SCHEME_ID;
use super::sample::estimate_compression_ratio_with_sampling;
use super::select::WinnerEstimate;
use super::structural;
use crate::builtins::FloatDictScheme;
use crate::builtins::IntDictScheme;
use crate::builtins::StringDictScheme;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
use crate::scheme::EstimateScore;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

fn compressor() -> CascadingCompressor {
    CascadingCompressor::new(vec![&IntDictScheme, &FloatDictScheme, &StringDictScheme])
}

fn estimate_test_data() -> ArrayAndStats {
    let array = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
    ArrayAndStats::new(array, GenerateStatsOptions::default())
}

fn matches_integer_primitive(canonical: &Canonical) -> bool {
    matches!(canonical, Canonical::Primitive(primitive) if primitive.ptype().is_int())
}

#[derive(Debug)]
struct DirectRatioScheme;

impl Scheme for DirectRatioScheme {
    fn scheme_name(&self) -> &'static str {
        "test.direct_ratio"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(2.0))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unreachable!("test helper should never be selected for compression")
    }
}

#[derive(Debug)]
struct ImmediateAlwaysUseScheme;

impl Scheme for ImmediateAlwaysUseScheme {
    fn scheme_name(&self) -> &'static str {
        "test.immediate_always_use"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
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
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unreachable!("test helper should never be selected for compression")
    }
}

#[derive(Debug)]
struct CallbackAlwaysUseScheme;

impl Scheme for CallbackAlwaysUseScheme {
    fn scheme_name(&self) -> &'static str {
        "test.callback_always_use"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, _data, _ctx, _exec_ctx, _best_so_far| Ok(EstimateVerdict::AlwaysUse),
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unreachable!("test helper should never be selected for compression")
    }
}

#[derive(Debug)]
struct CallbackSkipScheme;

impl Scheme for CallbackSkipScheme {
    fn scheme_name(&self) -> &'static str {
        "test.callback_skip"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, _data, _ctx, _exec_ctx, _best_so_far| Ok(EstimateVerdict::Skip),
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unreachable!("test helper should never be selected for compression")
    }
}

#[derive(Debug)]
struct CallbackRatioScheme;

impl Scheme for CallbackRatioScheme {
    fn scheme_name(&self) -> &'static str {
        "test.callback_ratio"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, _data, _ctx, _exec_ctx, _best_so_far| Ok(EstimateVerdict::Ratio(3.0)),
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unreachable!("test helper should never be selected for compression")
    }
}

#[derive(Debug)]
struct HugeRatioScheme;

impl Scheme for HugeRatioScheme {
    fn scheme_name(&self) -> &'static str {
        "test.huge_ratio"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(100.0))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unreachable!("test helper should never be selected for compression")
    }
}

#[derive(Debug)]
struct ZeroBytesSamplingScheme;

impl Scheme for ZeroBytesSamplingScheme {
    fn scheme_name(&self) -> &'static str {
        "test.zero_bytes_sampling"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(NullArray::new(data.array().len()).into_array())
    }
}

#[test]
fn test_self_exclusion() {
    let c = compressor();
    let ctx = CompressorContext::default().descend_with_scheme(IntDictScheme.id(), 0);

    // IntDictScheme is in the history, so it should be excluded.
    assert!(c.is_excluded(&IntDictScheme, &ctx));
}

#[test]
fn test_root_exclusion_list_offsets() {
    let c = compressor();
    let ctx = CompressorContext::default()
        .descend_with_scheme(ROOT_SCHEME_ID, structural::root_list_children::OFFSETS);

    // IntDict should be excluded for list offsets.
    assert!(c.is_excluded(&IntDictScheme, &ctx));
}

#[test]
fn test_push_rule_float_dict_excludes_int_dict_from_codes() {
    let c = compressor();
    // FloatDict cascading through codes (child 1).
    let ctx = CompressorContext::default().descend_with_scheme(FloatDictScheme.id(), 1);

    // IntDict should be excluded from FloatDict's codes child.
    assert!(c.is_excluded(&IntDictScheme, &ctx));
}

#[test]
fn test_push_rule_float_dict_excludes_int_dict_from_values() {
    let c = compressor();
    // FloatDict cascading through values (child 0).
    let ctx = CompressorContext::default().descend_with_scheme(FloatDictScheme.id(), 0);

    // IntDict should also be excluded from FloatDict's values child (ALP propagation
    // replacement).
    assert!(c.is_excluded(&IntDictScheme, &ctx));
}

#[test]
fn test_no_exclusion_without_history() {
    let c = compressor();
    let ctx = CompressorContext::default();

    // No history means no exclusions.
    assert!(!c.is_excluded(&IntDictScheme, &ctx));
}

#[test]
fn immediate_always_use_wins_immediately() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&DirectRatioScheme, &ImmediateAlwaysUseScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &ImmediateAlwaysUseScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, WinnerEstimate::AlwaysUse))
            if scheme.id() == ImmediateAlwaysUseScheme.id()
    ));
    Ok(())
}

#[test]
fn callback_always_use_wins_immediately() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&DirectRatioScheme, &CallbackAlwaysUseScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &CallbackAlwaysUseScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, WinnerEstimate::AlwaysUse))
            if scheme.id() == CallbackAlwaysUseScheme.id()
    ));
    Ok(())
}

#[test]
fn callback_skip_is_ignored() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&CallbackSkipScheme, &DirectRatioScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&CallbackSkipScheme, &DirectRatioScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, WinnerEstimate::Score { score: EstimateScore::FiniteCompression(2.0), .. }))
            if scheme.id() == DirectRatioScheme.id()
    ));
    Ok(())
}

#[test]
fn callback_ratio_competes_numerically() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&DirectRatioScheme, &CallbackRatioScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &CallbackRatioScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, WinnerEstimate::Score { score: EstimateScore::FiniteCompression(3.0), .. }))
            if scheme.id() == CallbackRatioScheme.id()
    ));
    Ok(())
}

#[test]
fn zero_byte_sample_loses_to_finite_ratio() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&HugeRatioScheme, &ZeroBytesSamplingScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&HugeRatioScheme, &ZeroBytesSamplingScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, WinnerEstimate::Score { score: EstimateScore::FiniteCompression(100.0), .. }))
            if scheme.id() == HugeRatioScheme.id()
    ));
    Ok(())
}

#[test]
fn finite_ratio_displaces_zero_byte_sample() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&ZeroBytesSamplingScheme, &HugeRatioScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&ZeroBytesSamplingScheme, &HugeRatioScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, WinnerEstimate::Score { score: EstimateScore::FiniteCompression(100.0), .. }))
            if scheme.id() == HugeRatioScheme.id()
    ));
    Ok(())
}

#[test]
fn zero_byte_sample_alone_selects_no_scheme() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&ZeroBytesSamplingScheme]);
    let schemes: [&'static dyn Scheme; 1] = [&ZeroBytesSamplingScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(winner.is_none());
    Ok(())
}

// Observer helper used by threshold-related tests. Captures the `best_so_far` value the
// compressor passes to its deferred callback. `OBSERVER_LOCK` serializes tests that share
// `OBSERVED_THRESHOLD` so they do not race.
static OBSERVER_LOCK: Mutex<()> = Mutex::new(());
static OBSERVED_THRESHOLD: Mutex<Option<Option<EstimateScore>>> = Mutex::new(None);

#[derive(Debug)]
struct ThresholdObservingScheme;

impl Scheme for ThresholdObservingScheme {
    fn scheme_name(&self) -> &'static str {
        "test.threshold_observing"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, _data, best_so_far, _ctx, _exec_ctx| {
                *OBSERVED_THRESHOLD.lock() = Some(best_so_far);
                Ok(EstimateVerdict::Skip)
            },
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unreachable!("test helper should never be selected for compression")
    }
}

#[derive(Debug)]
struct CallbackMatchingRatioScheme;

impl Scheme for CallbackMatchingRatioScheme {
    fn scheme_name(&self) -> &'static str {
        "test.callback_matching_ratio"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, _data, _ctx, _exec_ctx, _best_so_far| Ok(EstimateVerdict::Ratio(2.0)),
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unreachable!("test helper should never be selected for compression")
    }
}

#[test]
fn callback_always_use_overrides_pass_one_best() -> VortexResult<()> {
    // `HugeRatioScheme` returns an immediate `Ratio(100.0)` in pass 1;
    // `CallbackAlwaysUseScheme` returns `AlwaysUse` from its deferred callback in pass 2.
    // The deferred `AlwaysUse` must still win.
    let compressor = CascadingCompressor::new(vec![&HugeRatioScheme, &CallbackAlwaysUseScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&HugeRatioScheme, &CallbackAlwaysUseScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, WinnerEstimate::AlwaysUse))
            if scheme.id() == CallbackAlwaysUseScheme.id()
    ));
    Ok(())
}

#[test]
fn threshold_reflects_pass_one_best() -> VortexResult<()> {
    let _guard = OBSERVER_LOCK.lock();
    *OBSERVED_THRESHOLD.lock() = None;

    let compressor = CascadingCompressor::new(vec![&DirectRatioScheme, &ThresholdObservingScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &ThresholdObservingScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    let observed = *OBSERVED_THRESHOLD.lock();
    assert!(matches!(
        observed,
        Some(Some(EstimateScore::FiniteCompression(r))) if r == 2.0
    ));
    Ok(())
}

#[test]
fn threshold_is_none_when_only_prior_is_zero_bytes() -> VortexResult<()> {
    let _guard = OBSERVER_LOCK.lock();
    *OBSERVED_THRESHOLD.lock() = None;

    let compressor =
        CascadingCompressor::new(vec![&ZeroBytesSamplingScheme, &ThresholdObservingScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&ZeroBytesSamplingScheme, &ThresholdObservingScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    // The observing callback was invoked (outer `Some`) and `best_so_far` was `None` (inner
    // `None`) because the zero-byte sample is never stored as the best.
    let observed = *OBSERVED_THRESHOLD.lock();
    assert_eq!(observed, Some(None));
    Ok(())
}

#[test]
fn threshold_is_none_when_no_prior_scheme() -> VortexResult<()> {
    let _guard = OBSERVER_LOCK.lock();
    *OBSERVED_THRESHOLD.lock() = None;

    let compressor = CascadingCompressor::new(vec![&ThresholdObservingScheme]);
    let schemes: [&'static dyn Scheme; 1] = [&ThresholdObservingScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    let observed = *OBSERVED_THRESHOLD.lock();
    assert_eq!(observed, Some(None));
    Ok(())
}

#[test]
fn threshold_updates_from_earlier_deferred_callback() -> VortexResult<()> {
    let _guard = OBSERVER_LOCK.lock();
    *OBSERVED_THRESHOLD.lock() = None;

    // Both schemes are deferred. The first callback registers `Ratio(3.0)`; the second
    // callback must observe it as its threshold.
    let compressor =
        CascadingCompressor::new(vec![&CallbackRatioScheme, &ThresholdObservingScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&CallbackRatioScheme, &ThresholdObservingScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    let observed = *OBSERVED_THRESHOLD.lock();
    assert!(matches!(
        observed,
        Some(Some(EstimateScore::FiniteCompression(r))) if r == 3.0
    ));
    Ok(())
}

#[test]
fn ratio_tie_between_immediate_and_deferred_favors_immediate() -> VortexResult<()> {
    // Both schemes produce the same `Ratio(2.0)`, one from pass 1 (immediate) and one from
    // pass 2 (deferred callback). Pass 1 locks in first, and strict `>` tie-breaking means
    // the deferred callback's equal ratio cannot displace it.
    let compressor =
        CascadingCompressor::new(vec![&CallbackMatchingRatioScheme, &DirectRatioScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&CallbackMatchingRatioScheme, &DirectRatioScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, WinnerEstimate::Score { score: EstimateScore::FiniteCompression(r), .. }))
            if scheme.id() == DirectRatioScheme.id() && r == 2.0
    ));
    Ok(())
}

#[test]
fn all_null_array_compresses_to_constant() -> VortexResult<()> {
    let array = PrimitiveArray::new(
        buffer![0i32, 0, 0, 0, 0],
        Validity::Array(BoolArray::from_iter([false, false, false, false, false]).into_array()),
    )
    .into_array();

    // The compressor should produce a `ConstantArray` for an all-null array regardless of
    // which schemes are registered.
    let compressor = CascadingCompressor::new(vec![&IntDictScheme]);
    let mut exec_ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&array, &mut exec_ctx)?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

/// Regression test for <https://github.com/vortex-data/vortex/issues/7227>.
///
/// `estimate_compression_ratio_with_sampling` must use the *scheme's* stats options
/// (which request distinct-value counting) rather than the context's stats options
/// (which may not). With the old code this panicked inside `dictionary_encode` because
/// distinct values were never computed for the sample.
#[test]
fn sampling_uses_scheme_stats_options() -> VortexResult<()> {
    // Low-cardinality float array so FloatDictScheme considers it compressible.
    let array = PrimitiveArray::new(
        buffer![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        Validity::NonNullable,
    )
    .into_array();

    let compressor = CascadingCompressor::new(vec![&FloatDictScheme]);

    // A context with default stats_options (count_distinct_values = false) and
    // marked as a sample so the function skips the sampling step and compresses
    // the array directly.
    let ctx = CompressorContext::new().with_sampling();

    // Before the fix this panicked with:
    //   "this must be present since `DictScheme` declared that we need distinct values"
    let mut exec_ctx = SESSION.create_execution_ctx();
    let estimate = estimate_compression_ratio_with_sampling(
        &compressor,
        &FloatDictScheme,
        &array,
        ctx,
        &mut exec_ctx,
    )?;
    assert!(matches!(estimate.score, EstimateScore::FiniteCompression(ratio) if ratio.is_finite()));
    Ok(())
}
