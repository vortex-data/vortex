// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

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
use super::sample::evaluate_candidate_with_sampling;
use super::select::SelectionOutcome;
use super::structural;
use crate::builtins::FloatDictScheme;
use crate::builtins::IntDictScheme;
use crate::builtins::StringDictScheme;
use crate::cost::Candidate;
use crate::cost::Cost;
use crate::cost::CostModel;
use crate::scheme::CandidateEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEvaluation;
use crate::scheme::ResolvedEvaluation;
use crate::scheme::Scheme;
use crate::scheme::SchemeEvaluation;
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

    fn evaluate(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> SchemeEvaluation {
        SchemeEvaluation::Candidate(CandidateEstimate::from_compression_ratio(2.0))
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

    fn evaluate(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> SchemeEvaluation {
        SchemeEvaluation::AlwaysUse
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

    fn evaluate(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> SchemeEvaluation {
        SchemeEvaluation::Deferred(DeferredEvaluation::Callback(Box::new(
            |_compressor, _data, _ctx, _exec_ctx| Ok(ResolvedEvaluation::AlwaysUse),
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

    fn evaluate(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> SchemeEvaluation {
        SchemeEvaluation::Deferred(DeferredEvaluation::Callback(Box::new(
            |_compressor, _data, _ctx, _exec_ctx| Ok(ResolvedEvaluation::Skip),
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

    fn evaluate(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> SchemeEvaluation {
        SchemeEvaluation::Deferred(DeferredEvaluation::Callback(Box::new(
            |_compressor, _data, _ctx, _exec_ctx| {
                Ok(ResolvedEvaluation::Candidate(
                    CandidateEstimate::from_compression_ratio(3.0),
                ))
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
struct DeferredLowerRatioScheme;

impl Scheme for DeferredLowerRatioScheme {
    fn scheme_name(&self) -> &'static str {
        "test.deferred_lower_ratio"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn evaluate(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> SchemeEvaluation {
        SchemeEvaluation::Deferred(DeferredEvaluation::Callback(Box::new(
            |_compressor, _data, _compress_ctx, _exec_ctx| {
                Ok(ResolvedEvaluation::Candidate(
                    CandidateEstimate::from_compression_ratio(1.5),
                ))
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
struct HugeRatioScheme;

impl Scheme for HugeRatioScheme {
    fn scheme_name(&self) -> &'static str {
        "test.huge_ratio"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn evaluate(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> SchemeEvaluation {
        SchemeEvaluation::Candidate(CandidateEstimate::from_compression_ratio(100.0))
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

    fn evaluate(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> SchemeEvaluation {
        SchemeEvaluation::Deferred(DeferredEvaluation::Sample)
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

/// Assigns the lower-ratio direct scheme a lower cost than the high-ratio scheme, proving that
/// selector policy is supplied by the configured model rather than hard-coded ratio ordering.
#[derive(Debug)]
struct PreferDirectCost;

impl CostModel for PreferDirectCost {
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost> {
        if candidate.scheme_id() == DirectRatioScheme.id() {
            Some(Cost::new(0.0))
        } else {
            Some(Cost::new(1.0))
        }
    }

    fn canonical_cost(&self, _data: &ArrayAndStats, _n_values: u64) -> Cost {
        Cost::new(2.0)
    }
}

#[derive(Debug)]
struct PreferDeferredLowerRatioCost;

impl CostModel for PreferDeferredLowerRatioCost {
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost> {
        if candidate.scheme_id() == DeferredLowerRatioScheme.id() {
            Some(Cost::new(0.0))
        } else {
            Some(Cost::new(1.0))
        }
    }

    fn canonical_cost(&self, _data: &ArrayAndStats, _n_values: u64) -> Cost {
        Cost::new(2.0)
    }
}

#[derive(Debug)]
struct AboveCanonicalCost;

impl CostModel for AboveCanonicalCost {
    fn cost(&self, _candidate: &Candidate<'_>) -> Option<Cost> {
        Some(Cost::new(1.0))
    }

    fn canonical_cost(&self, _data: &ArrayAndStats, _n_values: u64) -> Cost {
        Cost::new(0.0)
    }
}

#[derive(Debug)]
struct ObservingCost {
    immediate_without_sample: Arc<AtomicBool>,
    sampled_with_sample: Arc<AtomicBool>,
}

impl CostModel for ObservingCost {
    fn cost(&self, candidate: &Candidate<'_>) -> Option<Cost> {
        assert_eq!(candidate.array().len(), 4);
        assert_eq!(candidate.n_values(), 4);
        assert_eq!(candidate.input_nbytes(), candidate.array().nbytes());
        assert!(candidate.cascade().is_empty());

        if candidate.scheme_id() == DirectRatioScheme.id() {
            self.immediate_without_sample.store(
                candidate.sampled().is_none()
                    && candidate.estimate().estimated_compression_ratio() == Some(2.0),
                Ordering::Relaxed,
            );
        } else if candidate.scheme_id() == ZeroBytesSamplingScheme.id() {
            self.sampled_with_sample.store(
                candidate.sampled().is_some()
                    && candidate.estimate().estimated_compression_ratio().is_none(),
                Ordering::Relaxed,
            );
        }

        None
    }

    fn canonical_cost(&self, _data: &ArrayAndStats, _n_values: u64) -> Cost {
        Cost::new(0.0)
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
        Some((scheme, SelectionOutcome::AlwaysUse))
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
        Some((scheme, SelectionOutcome::AlwaysUse))
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
        Some((scheme, SelectionOutcome::Cost(_)))
            if scheme.id() == DirectRatioScheme.id()
    ));
    Ok(())
}

#[test]
fn deferred_candidate_competes_by_cost() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&DirectRatioScheme, &CallbackRatioScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &CallbackRatioScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, SelectionOutcome::Cost(_)))
            if scheme.id() == CallbackRatioScheme.id()
    ));
    Ok(())
}

#[test]
fn custom_cost_model_changes_the_winner() -> VortexResult<()> {
    let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &HugeRatioScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let size_winner = CascadingCompressor::new(schemes.to_vec()).choose_best_scheme(
        &schemes,
        &data,
        CompressorContext::new(),
        &mut exec_ctx,
    )?;
    assert!(matches!(
        size_winner,
        Some((scheme, _)) if scheme.id() == HugeRatioScheme.id()
    ));

    let custom_winner = CascadingCompressor::new(schemes.to_vec())
        .with_cost_model(Arc::new(PreferDirectCost))
        .choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;
    assert!(matches!(
        custom_winner,
        Some((scheme, _)) if scheme.id() == DirectRatioScheme.id()
    ));
    Ok(())
}

#[test]
fn custom_model_can_choose_a_deferred_lower_ratio_candidate() -> VortexResult<()> {
    let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &DeferredLowerRatioScheme];
    let compressor = CascadingCompressor::new(schemes.to_vec())
        .with_cost_model(Arc::new(PreferDeferredLowerRatioCost));
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, _)) if scheme.id() == DeferredLowerRatioScheme.id()
    ));
    Ok(())
}

#[test]
fn candidate_must_beat_canonical_cost() -> VortexResult<()> {
    let schemes: [&'static dyn Scheme; 1] = [&DirectRatioScheme];
    let compressor =
        CascadingCompressor::new(schemes.to_vec()).with_cost_model(Arc::new(AboveCanonicalCost));
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(winner.is_none());
    Ok(())
}

#[test]
fn sampled_array_is_exposed_only_for_sampled_candidates() -> VortexResult<()> {
    let immediate_without_sample = Arc::new(AtomicBool::new(false));
    let sampled_with_sample = Arc::new(AtomicBool::new(false));
    let model = ObservingCost {
        immediate_without_sample: Arc::clone(&immediate_without_sample),
        sampled_with_sample: Arc::clone(&sampled_with_sample),
    };
    let schemes: [&'static dyn Scheme; 2] = [&DirectRatioScheme, &ZeroBytesSamplingScheme];
    let compressor = CascadingCompressor::new(schemes.to_vec()).with_cost_model(Arc::new(model));
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(winner.is_none());
    assert!(immediate_without_sample.load(Ordering::Relaxed));
    assert!(sampled_with_sample.load(Ordering::Relaxed));
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
        Some((scheme, SelectionOutcome::Cost(_)))
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
        Some((scheme, SelectionOutcome::Cost(_)))
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

#[derive(Debug)]
struct CallbackMatchingRatioScheme;

impl Scheme for CallbackMatchingRatioScheme {
    fn scheme_name(&self) -> &'static str {
        "test.callback_matching_ratio"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn evaluate(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> SchemeEvaluation {
        SchemeEvaluation::Deferred(DeferredEvaluation::Callback(Box::new(
            |_compressor, _data, _ctx, _exec_ctx| {
                Ok(ResolvedEvaluation::Candidate(
                    CandidateEstimate::from_compression_ratio(2.0),
                ))
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

#[test]
fn callback_always_use_overrides_pass_one_best() -> VortexResult<()> {
    // `HugeRatioScheme` returns an immediate candidate in pass 1;
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
        Some((scheme, SelectionOutcome::AlwaysUse))
            if scheme.id() == CallbackAlwaysUseScheme.id()
    ));
    Ok(())
}

#[test]
fn equal_cost_between_immediate_and_deferred_favors_immediate() -> VortexResult<()> {
    // Both schemes produce candidates with the same SizeCost, one from pass 1 and one from
    // pass 2. Pass 1 locks in first, and strict cost comparison keeps that candidate.
    let compressor =
        CascadingCompressor::new(vec![&CallbackMatchingRatioScheme, &DirectRatioScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&CallbackMatchingRatioScheme, &DirectRatioScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, SelectionOutcome::Cost(_)))
            if scheme.id() == DirectRatioScheme.id()
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
/// `evaluate_candidate_with_sampling` must use the *scheme's* stats options
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
    let candidate = evaluate_candidate_with_sampling(
        &compressor,
        &FloatDictScheme,
        &array,
        ctx,
        &mut exec_ctx,
    )?;
    assert!(
        candidate
            .estimate
            .estimated_compression_ratio()
            .is_some_and(f64::is_finite)
    );
    Ok(())
}
