// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::Map;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::Variant;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveSlots;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::MapBuilder;
use vortex_array::builders::dict::dict_encode;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnionVariants;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::CascadingCompressor;
use super::ROOT_SCHEME_ID;
use super::TrialOutput;
use super::sample::estimate_compression_ratio_with_sampling;
use super::select::WinnerEstimate;
use super::structural;
use crate::builtins::FloatDictScheme;
use crate::builtins::IntDictScheme;
use crate::builtins::StringDictScheme;
use crate::scheme::Child;
use crate::scheme::ChildDecision;
use crate::scheme::ChildKind;
use crate::scheme::ChildPolicy;
use crate::scheme::ChildRequest;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
use crate::scheme::Encoded;
use crate::scheme::EstimateScore;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
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

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
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

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
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

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
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

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
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

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
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

/// Samples to a finite ratio by writing a single byte.
#[derive(Debug)]
struct FiniteSamplingScheme;

impl Scheme for FiniteSamplingScheme {
    fn scheme_name(&self) -> &'static str {
        "test.finite_sampling"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
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
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::new(buffer![0u8], Validity::NonNullable).into_array())
    }
}

/// An `AlwaysUse` default that samples to a finite ratio by writing a single byte.
#[derive(Debug)]
struct SampledAlwaysUseScheme;

impl Scheme for SampledAlwaysUseScheme {
    fn scheme_name(&self) -> &'static str {
        "test.sampled_always_use"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
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
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::new(buffer![0u8], Validity::NonNullable).into_array())
    }
}

/// An `AlwaysUse` default whose samples compress to zero bytes, as samples of only nulls do.
#[derive(Debug)]
struct ZeroBytesAlwaysUseScheme;

impl Scheme for ZeroBytesAlwaysUseScheme {
    fn scheme_name(&self) -> &'static str {
        "test.zero_bytes_always_use"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
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
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(NullArray::new(data.array().len()).into_array())
    }
}

/// Promises a ratio it does not deliver: its output is the input.
#[derive(Debug)]
struct OverpromisingScheme;

impl Scheme for OverpromisingScheme {
    fn scheme_name(&self) -> &'static str {
        "test.overpromising"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
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
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(100.0))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(data.array().clone())
    }
}

/// Stores a dictionary as its trial output, then reports that it cannot beat the threshold.
#[derive(Debug)]
struct TrialStoringScheme;

impl Scheme for TrialStoringScheme {
    fn scheme_name(&self) -> &'static str {
        "test.trial_storing"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
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
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, _best_so_far, _ctx, exec_ctx| {
                let array = dict_encode(data.array(), exec_ctx)?.into_array();
                data.get_or_insert_with(|| TrialOutput {
                    scheme: TrialStoringScheme.id(),
                    array,
                });
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

#[test]
fn test_self_exclusion() {
    let c = compressor();
    let ctx = CompressorContext::new().descend_with_scheme(IntDictScheme.id(), 0);

    // IntDictScheme is in the history, so it should be excluded.
    assert!(c.is_excluded(&IntDictScheme, &ctx));
}

#[test]
fn test_root_exclusion_list_offsets() {
    let c = compressor();
    let ctx = CompressorContext::new()
        .descend_with_scheme(ROOT_SCHEME_ID, structural::root_list_children::OFFSETS);

    // IntDict should be excluded for list offsets.
    assert!(c.is_excluded(&IntDictScheme, &ctx));
}

#[test]
fn test_push_rule_float_dict_excludes_int_dict_from_codes() {
    let c = compressor();
    // FloatDict cascading through codes (child 1).
    let ctx = CompressorContext::new().descend_with_scheme(FloatDictScheme.id(), 1);

    // IntDict should be excluded from FloatDict's codes child.
    assert!(c.is_excluded(&IntDictScheme, &ctx));
}

#[test]
fn test_push_rule_float_dict_excludes_int_dict_from_values() {
    let c = compressor();
    // FloatDict cascading through values (child 0).
    let ctx = CompressorContext::new().descend_with_scheme(FloatDictScheme.id(), 0);

    // IntDict should also be excluded from FloatDict's values child (ALP propagation
    // replacement).
    assert!(c.is_excluded(&IntDictScheme, &ctx));
}

#[test]
fn test_no_exclusion_without_history() {
    let c = compressor();
    let ctx = CompressorContext::new();

    // No history means no exclusions.
    assert!(!c.is_excluded(&IntDictScheme, &ctx));
}

/// Without other candidates, the default wins without being sampled: its `compress` panics.
#[test]
fn always_use_wins_immediately_without_competitors() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&ImmediateAlwaysUseScheme]);
    let schemes: [&'static dyn Scheme; 1] = [&ImmediateAlwaysUseScheme];
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

/// The default competes on a sample, and a candidate with a better estimate beats it.
#[rstest]
#[case::default_wins(&DirectRatioScheme, SampledAlwaysUseScheme.id())]
#[case::candidate_wins(&HugeRatioScheme, HugeRatioScheme.id())]
fn always_use_default_competes_with_candidates(
    #[case] candidate: &'static dyn Scheme,
    #[case] expected: SchemeId,
) -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![candidate, &SampledAlwaysUseScheme]);
    let schemes: [&'static dyn Scheme; 2] = [candidate, &SampledAlwaysUseScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, WinnerEstimate::Score(_))) if scheme.id() == expected
    ));
    Ok(())
}

/// When no candidate has a valid estimate, e.g. every sample holds only nulls, the default is
/// chosen rather than no scheme.
#[test]
fn always_use_default_is_chosen_without_valid_estimates() -> VortexResult<()> {
    let compressor =
        CascadingCompressor::new(vec![&ZeroBytesSamplingScheme, &ZeroBytesAlwaysUseScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&ZeroBytesSamplingScheme, &ZeroBytesAlwaysUseScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let winner =
        compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    assert!(matches!(
        winner,
        Some((scheme, WinnerEstimate::AlwaysUse))
            if scheme.id() == ZeroBytesAlwaysUseScheme.id()
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
        Some((scheme, WinnerEstimate::Score(EstimateScore::FiniteCompression(2.0))))
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
        Some((scheme, WinnerEstimate::Score(EstimateScore::FiniteCompression(3.0))))
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
        Some((scheme, WinnerEstimate::Score(EstimateScore::FiniteCompression(100.0))))
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
        Some((scheme, WinnerEstimate::Score(EstimateScore::FiniteCompression(100.0))))
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

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
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

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
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
fn callbacks_observe_samples_registered_after_them() -> VortexResult<()> {
    let _guard = OBSERVER_LOCK.lock();
    *OBSERVED_THRESHOLD.lock() = None;

    let compressor =
        CascadingCompressor::new(vec![&ThresholdObservingScheme, &FiniteSamplingScheme]);
    let schemes: [&'static dyn Scheme; 2] = [&ThresholdObservingScheme, &FiniteSamplingScheme];
    let data = estimate_test_data();
    let mut exec_ctx = SESSION.create_execution_ctx();

    compressor.choose_best_scheme(&schemes, &data, CompressorContext::new(), &mut exec_ctx)?;

    let observed = *OBSERVED_THRESHOLD.lock();
    assert!(matches!(
        observed,
        Some(Some(EstimateScore::FiniteCompression(_)))
    ));
    Ok(())
}

#[test]
fn smaller_trial_output_replaces_an_overpromising_winner() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&OverpromisingScheme, &TrialStoringScheme]);
    let array = PrimitiveArray::from_iter((0..1024i64).map(|value| value % 4)).into_array();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let compressed = compressor.compress(&array, &mut exec_ctx)?;

    assert!(compressed.is::<Dict>());
    assert_arrays_eq!(compressed, array, &mut exec_ctx);
    Ok(())
}

#[test]
fn all_null_sample_gives_no_estimate() -> VortexResult<()> {
    // Only the first row is valid, and the sampler does not land on it.
    let array =
        PrimitiveArray::from_option_iter((0..100_000).map(|index| (index == 0).then_some(index)))
            .into_array();
    let compressor = CascadingCompressor::new(vec![&FiniteSamplingScheme]);
    let mut exec_ctx = SESSION.create_execution_ctx();

    let score = estimate_compression_ratio_with_sampling(
        &compressor,
        &FiniteSamplingScheme,
        &ArrayAndStats::new(array, GenerateStatsOptions::default()),
        CompressorContext::new(),
        &mut exec_ctx,
    )?;

    assert_eq!(score, EstimateScore::ZeroBytes);
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
        Some((scheme, WinnerEstimate::Score(EstimateScore::FiniteCompression(r))))
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
    assert_arrays_eq!(compressed, array, &mut exec_ctx);
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
    let score = estimate_compression_ratio_with_sampling(
        &compressor,
        &FloatDictScheme,
        &ArrayAndStats::new(array, GenerateStatsOptions::default()),
        ctx,
        &mut exec_ctx,
    )?;
    assert!(matches!(score, EstimateScore::FiniteCompression(ratio) if ratio.is_finite()));
    Ok(())
}

/// Compresses every `i32` row to one byte, after a table of [`FixedTableScheme::TABLE_NBYTES`]
/// bytes that stays the same size however many rows there are.
#[derive(Debug)]
struct FixedTableScheme;

impl FixedTableScheme {
    const TABLE_NBYTES: usize = 4096;
}

impl Scheme for FixedTableScheme {
    fn scheme_name(&self) -> &'static str {
        "test.fixed_table"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
    }

    fn fixed_nbytes(&self, _compressed: &ArrayRef) -> u64 {
        Self::TABLE_NBYTES as u64
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
        let nbytes = Self::TABLE_NBYTES + data.array().len();
        Ok(PrimitiveArray::new(
            Buffer::from_iter(iter::repeat_n(0u8, nbytes)),
            Validity::NonNullable,
        )
        .into_array())
    }
}

/// A sample pays for a fixed-size table in full, which would rank [`FixedTableScheme`] below the
/// canonical encoding (4096 bytes in, 5120 out) although it shrinks the full array almost 4x.
#[test]
fn sampling_amortizes_fixed_nbytes_over_the_full_array() -> VortexResult<()> {
    let array =
        PrimitiveArray::new(Buffer::from_iter(0..102_400i32), Validity::NonNullable).into_array();
    let compressor = CascadingCompressor::new(vec![&FixedTableScheme]);
    let mut exec_ctx = SESSION.create_execution_ctx();

    let score = estimate_compression_ratio_with_sampling(
        &compressor,
        &FixedTableScheme,
        &ArrayAndStats::new(array, GenerateStatsOptions::default()),
        CompressorContext::new(),
        &mut exec_ctx,
    )?;

    // The 1024 sampled rows are 1% of the array, so they carry 1% of the table, rounded up.
    assert_eq!(score, EstimateScore::from_sample_sizes(4 * 1024, 1024 + 41));
    Ok(())
}

/// An array that is compressed whole has nothing to amortize the table over.
#[test]
fn sampling_charges_fixed_nbytes_in_full_without_subsampling() -> VortexResult<()> {
    let array =
        PrimitiveArray::new(Buffer::from_iter(0..8192i32), Validity::NonNullable).into_array();
    let compressor = CascadingCompressor::new(vec![&FixedTableScheme]);
    let mut exec_ctx = SESSION.create_execution_ctx();

    let score = estimate_compression_ratio_with_sampling(
        &compressor,
        &FixedTableScheme,
        &ArrayAndStats::new(array, GenerateStatsOptions::default()),
        CompressorContext::new().with_sampling(),
        &mut exec_ctx,
    )?;

    assert_eq!(
        score,
        EstimateScore::from_sample_sizes(4 * 8192, 4096 + 8192)
    );
    Ok(())
}

type MapEntryFixture<'a> = (i32, Option<&'a str>);
type MapRowFixture<'a> = Option<Vec<MapEntryFixture<'a>>>;

fn map_array_from_rows(rows: &[MapRowFixture<'_>], keys_sorted: bool) -> VortexResult<ArrayRef> {
    let map_dtype = MapDType::try_new(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable),
        keys_sorted,
    )?;
    let dtype = DType::Map(map_dtype.clone(), Nullability::Nullable);
    let mut builder = MapBuilder::<u64, u64>::with_capacity_in(
        map_dtype,
        Nullability::Nullable,
        rows.len(),
        vortex_buffer::BufferAllocatorRef::static_ref(),
    );

    for row in rows {
        let scalar = match row {
            Some(entries) => {
                let entries = entries
                    .iter()
                    .map(|(key, value)| {
                        let key = Scalar::primitive(*key, Nullability::NonNullable);
                        let value = value.map_or_else(
                            || Scalar::null(DType::Utf8(Nullability::Nullable)),
                            |value| Scalar::utf8(value, Nullability::Nullable),
                        );
                        (key, value)
                    })
                    .collect::<Vec<_>>();
                Scalar::try_map(dtype.clone(), entries)?
            }
            None => Scalar::null(dtype.clone()),
        };
        builder.append_value(scalar.as_map())?;
    }

    Ok(builder.finish_into_map().into_array())
}

#[test]
fn map_compression_preserves_mixed_rows() -> VortexResult<()> {
    let array = map_array_from_rows(
        &[
            Some(vec![(1, Some("one")), (2, None)]),
            None,
            Some(vec![]),
            Some(vec![(1, Some("dup-old")), (1, Some("dup-new"))]),
        ],
        false,
    )?;
    let mut exec_ctx = SESSION.create_execution_ctx();

    let compressed = compressor().compress(&array, &mut exec_ctx)?;

    assert!(compressed.is::<Map>());
    assert_eq!(compressed.dtype(), array.dtype());
    assert_arrays_eq!(&compressed, &array, &mut exec_ctx);
    Ok(())
}

#[test]
fn all_null_map_compression_preserves_values() -> VortexResult<()> {
    let array = map_array_from_rows(&[None, None, None], false)?;
    let mut exec_ctx = SESSION.create_execution_ctx();

    let compressed = compressor().compress(&array, &mut exec_ctx)?;

    assert_eq!(compressed.dtype(), array.dtype());
    assert_arrays_eq!(&compressed, &array, &mut exec_ctx);
    Ok(())
}

#[test]
fn map_compression_preserves_repeated_entry_children() -> VortexResult<()> {
    let rows = (0..64)
        .map(|idx| {
            Some(vec![
                (idx % 4, Some("alpha")),
                (idx % 4, Some("beta")),
                (idx % 4, Some("alpha")),
            ])
        })
        .collect::<Vec<_>>();
    let array = map_array_from_rows(&rows, true)?;
    let mut exec_ctx = SESSION.create_execution_ctx();

    let compressed = compressor().compress(&array, &mut exec_ctx)?;

    assert!(compressed.is::<Map>());
    assert_eq!(compressed.dtype(), array.dtype());
    assert_arrays_eq!(&compressed, &array, &mut exec_ctx);
    Ok(())
}

/// Distinct 27-byte strings whose views reference a small prefix of one 2 MiB buffer, the way
/// Arrow's Parquet reader shares a decompressed page between batches.
fn views_into_shared_page(rows: u32) -> ArrayRef {
    const VALUE_LEN: u32 = 27;
    let mut page = vec![0u8; 2 << 20];
    for row in 0..rows {
        let value = format!("value-{row:021}");
        let start = (row * VALUE_LEN) as usize;
        page[start..start + VALUE_LEN as usize].copy_from_slice(value.as_bytes());
    }
    let page = ByteBuffer::from(page);
    let views = (0..rows)
        .map(|row| {
            let start = (row * VALUE_LEN) as usize;
            BinaryView::make_view(&page[start..start + VALUE_LEN as usize], 0, row * VALUE_LEN)
        })
        .collect::<Buffer<BinaryView>>();
    // SAFETY: every view points at `VALUE_LEN` bytes inside the page.
    unsafe {
        VarBinViewArray::new_unchecked(
            views,
            Arc::from([page]),
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        )
    }
    .into_array()
}

#[test]
fn scheme_must_beat_referenced_bytes_not_held_bytes() -> VortexResult<()> {
    let mut exec_ctx = SESSION.create_execution_ctx();
    let array = views_into_shared_page(2000);
    let exact = uncompressed_size_in_bytes(&array, &mut exec_ctx)?;

    // Every value is distinct, so a dictionary is larger than the referenced bytes but far
    // smaller than the 2 MiB the array holds. It must lose.
    let compressed = compressor().compress(&array, &mut exec_ctx)?;

    assert!(compressed.is::<VarBinView>());
    assert_eq!(compressed.nbytes(), exact as u64);
    assert_arrays_eq!(&compressed, &array, &mut exec_ctx);
    Ok(())
}

/// An extension type over fixed-size lists, like UUIDs over 16 bytes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct BytesExtension;

impl ExtVTable for BytesExtension {
    type Metadata = u8;
    type StoredMetadata = Self::Metadata;
    type NativeValue<'a> = &'a ScalarValue;

    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    fn id(&self) -> ExtId {
        ExtId::new("test.bytes")
    }

    fn serialize_metadata(&self, metadata: &u8) -> VortexResult<Vec<u8>> {
        Ok(vec![*metadata])
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::StoredMetadata> {
        vortex_ensure!(metadata.len() == 1, "expected one metadata byte");
        Ok(metadata[0])
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(stored_metadata)
    }

    fn storage_dtype(_metadata: &u8) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            16,
            Nullability::NonNullable,
        )
    }

    fn unpack_native<'a>(
        _dtype: &'a ExtDType<Self>,
        value: &'a ScalarValue,
    ) -> VortexResult<&'a ScalarValue> {
        Ok(value)
    }
}

/// Canonicalizing an extension array only canonicalizes its storage's top level, so the elements
/// of fixed-size list storage can stay a lazy slice. When no scheme compresses such an array it
/// must not be returned as-is, because the slice is not serializable.
#[test]
fn extension_with_lazy_storage_is_not_passed_through() -> VortexResult<()> {
    // Dictionaries of wide values keep the lazy array smaller than its canonical form, and
    // chunked arrays only slice on the execute path, so the slice stays lazy.
    let dict = |offset: u64| -> VortexResult<ArrayRef> {
        let codes = PrimitiveArray::from_iter((0..1024u32).map(|i| (i % 4) as u8)).into_array();
        let values = PrimitiveArray::from_iter([offset, offset + 1, offset + 2, offset + 3]);
        Ok(DictArray::try_new(codes, values.into_array())?.into_array())
    };
    let chunks = vec![dict(0)?, dict(100)?];
    let elements_dtype = chunks[0].dtype().clone();
    let chunked = ChunkedArray::try_new(chunks, elements_dtype)?.into_array();
    let elements = SliceArray::new(chunked, 16..2048).into_array();
    let storage = FixedSizeListArray::new(elements, 16, Validity::NonNullable, 127).into_array();
    let dtype = ExtDType::<BytesExtension>::try_new(0, storage.dtype().nullability())?.erased();
    let array = ExtensionArray::new(dtype, storage).into_array();
    let mut exec_ctx = SESSION.create_execution_ctx();

    let compressed = CascadingCompressor::new(Vec::new()).compress(&array, &mut exec_ctx)?;

    assert!(
        !compressed.depth_first_traversal().any(|a| a.is::<Slice>()),
        "compressed array still contains a lazy slice:\n{}",
        compressed.display_tree()
    );
    assert_arrays_eq!(&compressed, &array, &mut exec_ctx);
    Ok(())
}

/// Dictionary-encodes `[1, 2, 1, 2, ...]`, registering its values as `Encoded::keep` when
/// `keep_values`, or not registering them at all.
#[derive(Debug)]
struct TestDictScheme {
    name: &'static str,
    keep_values: bool,
}

static KEEPING_DICT: TestDictScheme = TestDictScheme {
    name: "test.keeping_dict",
    keep_values: true,
};

/// Only the debug-only check of registered children uses it.
#[cfg(debug_assertions)]
static LEAKY_DICT: TestDictScheme = TestDictScheme {
    name: "test.leaky_dict",
    keep_values: false,
};

impl Scheme for TestDictScheme {
    fn scheme_name(&self) -> &'static str {
        self.name
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
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

    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let codes = PrimitiveArray::new(
            [0u8, 1]
                .into_iter()
                .cycle()
                .take(data.array_len())
                .collect::<Buffer<u8>>(),
            Validity::NonNullable,
        )
        .into_array();
        let values = PrimitiveArray::new(buffer![1i32, 2], Validity::NonNullable).into_array();
        let keep = if self.keep_values {
            vec![Child::new(self, 1, values)]
        } else {
            Vec::new()
        };
        let keep_values = self.keep_values;
        Ok(Encoded::new(
            vec![Child::new(self, 0, codes)],
            keep,
            move |children, _| {
                let [codes] = children.compressed()?;
                let values = if keep_values {
                    let [values] = children.kept()?;
                    values
                } else {
                    PrimitiveArray::new(buffer![1i32, 2], Validity::NonNullable).into_array()
                };
                Ok(DictArray::try_new(codes, values)?.into_array())
            },
        ))
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

/// Records every child the compressor asks about, and answers with `decision`, or with what the
/// scheme requested when `None`.
#[derive(Debug, Default)]
struct RecordingPolicy {
    decision: Option<ChildDecision>,
    requests: Mutex<Vec<(ChildKind, ChildDecision)>>,
}

impl ChildPolicy for RecordingPolicy {
    fn decide(&self, request: &ChildRequest<'_>) -> ChildDecision {
        self.requests.lock().push((request.kind, request.requested));
        self.decision.unwrap_or(request.requested)
    }
}

fn alternating_ints() -> ArrayRef {
    PrimitiveArray::new(
        (0..4096).map(|i| 1 + i % 2).collect::<Buffer<i32>>(),
        Validity::NonNullable,
    )
    .into_array()
}

#[cfg(debug_assertions)]
#[test]
fn unregistered_child_is_rejected() {
    let compressor = CascadingCompressor::new(vec![&LEAKY_DICT]);
    let err = compressor
        .compress(&alternating_ints(), &mut SESSION.create_execution_ctx())
        .expect_err("the values child was never registered");
    assert!(
        err.to_string()
            .contains("test.leaky_dict left slot 1 (values) of vortex.dict uncompressed"),
        "{err}"
    );
}

#[test]
fn policy_sees_each_registered_child() -> VortexResult<()> {
    let policy = Arc::new(RecordingPolicy::default());
    let compressor =
        CascadingCompressor::new(vec![&KEEPING_DICT])
            .with_child_policy(Arc::<RecordingPolicy>::clone(&policy));
    let input = alternating_ints();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&input, &mut ctx)?;
    assert_arrays_eq!(&compressed, &input, &mut ctx);

    let cascade = |index| ChildKind::Cascade {
        scheme: KEEPING_DICT.id(),
        index,
    };
    assert_eq!(
        *policy.requests.lock(),
        [
            (cascade(0), ChildDecision::Compress),
            (cascade(1), ChildDecision::Keep),
        ]
    );
    Ok(())
}

#[test]
fn policy_can_compress_a_kept_child() -> VortexResult<()> {
    let policy = Arc::new(RecordingPolicy {
        decision: Some(ChildDecision::Compress),
        ..RecordingPolicy::default()
    });
    // With the kept values compressed, the policy is asked about the values' own children too,
    // and the result still round-trips.
    let compressor = CascadingCompressor::new(vec![&KEEPING_DICT]).with_child_policy(policy);
    let input = alternating_ints();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&input, &mut ctx)?;
    assert_arrays_eq!(&compressed, &input, &mut ctx);
    Ok(())
}

/// Implements only the original [`Scheme::compress`], compressing its own children with
/// [`CascadingCompressor::compress_child`].
#[derive(Debug)]
struct CompressOnlyDictScheme;

impl Scheme for CompressOnlyDictScheme {
    fn scheme_name(&self) -> &'static str {
        "test.compress_only_dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
    }

    fn num_children(&self) -> usize {
        2
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
        let dict = dict_encode(data.array(), exec_ctx)?;
        let values =
            compressor.compress_child(dict.values(), &compress_ctx, self.id(), 0, exec_ctx)?;
        let codes =
            compressor.compress_child(dict.codes(), &compress_ctx, self.id(), 1, exec_ctx)?;
        Ok(DictArray::try_new(codes, values)?.into_array())
    }
}

/// Leaves [`Scheme::compress`] unimplemented without implementing [`Scheme::encode`].
#[derive(Debug)]
struct NoEncodingScheme;

impl Scheme for NoEncodingScheme {
    fn scheme_name(&self) -> &'static str {
        "test.no_encoding"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
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

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

#[test]
fn compress_only_scheme_still_works() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&CompressOnlyDictScheme]);
    let input = PrimitiveArray::from_option_iter((0..4096).map(|i| (i % 7 != 0).then_some(i % 3)))
        .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&input, &mut ctx)?;
    assert!(compressed.is::<Dict>(), "{}", compressed.encoding_id());
    assert_arrays_eq!(&compressed, &input, &mut ctx);
    Ok(())
}

#[test]
#[should_panic(expected = "test.no_encoding compresses through Scheme::encode")]
fn scheme_without_encoding_panics() {
    let compressor = CascadingCompressor::new(vec![&NoEncodingScheme]);
    drop(compressor.compress(&alternating_ints(), &mut SESSION.create_execution_ctx()));
}

/// Registers its input as a kept child, then builds its output without it.
#[derive(Debug)]
struct UnusedChildScheme;

impl Scheme for UnusedChildScheme {
    fn scheme_name(&self) -> &'static str {
        "test.unused_child"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
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

    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let len = data.array_len();
        Ok(Encoded::new(
            Vec::new(),
            vec![Child::new(self, 0, data.array().clone())],
            move |_, _| Ok(ConstantArray::new(1i32, len).into_array()),
        ))
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

#[test]
fn unused_child_is_rejected() {
    let compressor = CascadingCompressor::new(vec![&UnusedChildScheme]);
    let err = compressor
        .compress(&alternating_ints(), &mut SESSION.create_execution_ctx())
        .expect_err("the kept child was never used");
    assert!(
        err.to_string().contains(
            "the constructor of scheme test.unused_child left 0 compressed and 1 kept children \
             unused"
        ),
        "{err}"
    );
}

/// Counts how often the compressor offers it a boolean array, and never matches.
#[derive(Debug)]
struct CountingBoolScheme;

static BOOL_ARRAYS_OFFERED: AtomicUsize = AtomicUsize::new(0);

impl Scheme for CountingBoolScheme {
    fn scheme_name(&self) -> &'static str {
        "test.counting_bool"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        if matches!(canonical, Canonical::Bool(_)) {
            BOOL_ARRAYS_OFFERED.fetch_add(1, Ordering::Relaxed);
        }
        false
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
        CompressionEstimate::Verdict(EstimateVerdict::Skip)
    }

    #[expect(clippy::unimplemented, reason = "never matches")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} never matches", self.scheme_name())
    }
}

/// Generates integer statistics while estimating, then declines.
#[derive(Debug)]
struct StatsThenSkipScheme;

impl Scheme for StatsThenSkipScheme {
    fn scheme_name(&self) -> &'static str {
        "test.stats_then_skip"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        data.integer_stats(exec_ctx);
        CompressionEstimate::Verdict(EstimateVerdict::Skip)
    }

    #[expect(clippy::unimplemented, reason = "always skips")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} always skips", self.scheme_name())
    }
}

/// Statistics generated while choosing a scheme show the leaf has no nulls, so its all-`true`
/// bitmap becomes a constant without going through the compressor again.
#[test]
fn all_valid_bitmap_reuses_statistics() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&StatsThenSkipScheme, &CountingBoolScheme]);
    let input = PrimitiveArray::new(
        (0..4096).collect::<Buffer<i32>>(),
        Validity::Array(BoolArray::from_iter(iter::repeat_n(true, 4096)).into_array()),
    )
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&input, &mut ctx)?;
    assert_arrays_eq!(&compressed, &input, &mut ctx);

    let bitmap = compressed.slots()[PrimitiveSlots::VALIDITY_CHILD]
        .as_ref()
        .ok_or_else(|| vortex_err!("expected the validity bitmap to stay a slot"))?;
    assert!(bitmap.is::<Constant>(), "{}", bitmap.encoding_id());
    assert_eq!(BOOL_ARRAYS_OFFERED.load(Ordering::Relaxed), 0);
    Ok(())
}

/// Hands its whole input to the compressor as its only child, and returns that child as it is.
#[derive(Debug)]
struct PassThroughScheme;

impl Scheme for PassThroughScheme {
    fn scheme_name(&self) -> &'static str {
        "test.pass_through"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
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

    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        Ok(Encoded::new(
            vec![Child::new(self, 0, data.array().clone())],
            Vec::new(),
            |children, _| {
                let [child] = children.compressed()?;
                Ok(child)
            },
        ))
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

/// A constructor may return a compressed child as its whole output: what the compressor left
/// canonical inside that child, here the dictionary's kept values, is not the scheme's to register.
#[test]
fn returning_a_compressed_child_passes_the_check() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&PassThroughScheme, &KEEPING_DICT]);
    let input = alternating_ints();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&input, &mut ctx)?;
    assert!(compressed.is::<Dict>(), "{}", compressed.encoding_id());
    assert_arrays_eq!(&compressed, &input, &mut ctx);
    Ok(())
}

#[derive(Debug)]
struct EmptyAllowlistParent;

impl Scheme for EmptyAllowlistParent {
    fn scheme_name(&self) -> &'static str {
        "test.empty_allowlist_parent"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Primitive(_))
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
    }

    fn num_children(&self) -> usize {
        1
    }

    fn descendant_allowlists(&self) -> Vec<crate::scheme::DescendantAllowlist> {
        vec![crate::scheme::DescendantAllowlist {
            allowed: Vec::new(),
            children: crate::scheme::ChildSelection::One(0),
        }]
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Verdict(EstimateVerdict::Skip)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(data.array().clone())
    }
}

/// A `DescendantAllowlist` with an empty `allowed` list must impose no restriction.
#[test]
fn empty_descendant_allowlist_does_not_exclude_schemes() {
    let compressor = CascadingCompressor::new(vec![&EmptyAllowlistParent, &KEEPING_DICT]);
    let root_ctx = CompressorContext::new();
    assert!(compressor.is_child_scheme_eligible(
        KEEPING_DICT.id(),
        &root_ctx,
        EmptyAllowlistParent.id(),
        0,
    ));
}

/// Leaf schemes remain eligible at `allowed_cascading == 0` (matching `choose_and_compress`),
/// while child cascading is still allowed at `allowed_cascading == 1` and blocked at `0`.
#[test]
fn test_scheme_eligibility_at_final_cascade_level() {
    let compressor = CascadingCompressor::new(vec![&EmptyAllowlistParent, &KEEPING_DICT]);
    let leaf_ctx = CompressorContext::new().as_leaf();
    assert!(leaf_ctx.finished_cascading());
    assert!(compressor.is_scheme_eligible(KEEPING_DICT.id(), &leaf_ctx));
    assert!(!compressor.is_child_scheme_eligible(
        KEEPING_DICT.id(),
        &leaf_ctx,
        EmptyAllowlistParent.id(),
        0,
    ));

    // One cascade level remaining (`allowed_cascading == 1`): descending yields `finished_cascading`,
    // yet the child level can still run leaf schemes.
    let one_left = (0..crate::scheme::MAX_CASCADE - 1).fold(CompressorContext::new(), |ctx, i| {
        ctx.descend_with_scheme(ROOT_SCHEME_ID, i)
    });
    assert!(!one_left.finished_cascading());
    assert!(
        one_left
            .clone()
            .descend_with_scheme(EmptyAllowlistParent.id(), 0)
            .finished_cascading()
    );
    assert!(compressor.is_child_scheme_eligible(
        KEEPING_DICT.id(),
        &one_left,
        EmptyAllowlistParent.id(),
        0,
    ));
}

/// When no scheme is eligible in a child context, `compress_child` returns a non-canonical child
/// as-is without canonicalizing it, while canonical children still run through `compress_canonical`.
#[test]
fn non_canonical_child_skipped_when_no_scheme_eligible() -> VortexResult<()> {
    #[derive(Debug)]
    struct AbsentAllowedScheme;

    impl Scheme for AbsentAllowedScheme {
        fn scheme_name(&self) -> &'static str {
            "test.absent_allowed"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches!(canonical, Canonical::Primitive(_))
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
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            Ok(data.array().clone())
        }
    }

    #[derive(Debug)]
    struct RestrictedAllowlistParent;

    impl Scheme for RestrictedAllowlistParent {
        fn scheme_name(&self) -> &'static str {
            "test.restricted_allowlist_parent"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            matches!(canonical, Canonical::Primitive(_))
        }

        fn produced_encodings(&self) -> Vec<ArrayId> {
            Vec::new()
        }

        fn num_children(&self) -> usize {
            1
        }

        fn descendant_allowlists(&self) -> Vec<crate::scheme::DescendantAllowlist> {
            vec![crate::scheme::DescendantAllowlist {
                allowed: vec![AbsentAllowedScheme.id()],
                children: crate::scheme::ChildSelection::One(0),
            }]
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            Ok(data.array().clone())
        }
    }

    let mut ctx = SESSION.create_execution_ctx();
    let dict_compressor = CascadingCompressor::new(vec![&KEEPING_DICT]);
    let dict_child = dict_compressor.compress(&alternating_ints(), &mut ctx)?;
    assert!(dict_child.is::<Dict>());

    let compressor = CascadingCompressor::new(vec![&RestrictedAllowlistParent, &KEEPING_DICT]);
    let root_ctx = CompressorContext::new();

    // Non-canonical child is returned unchanged when no registered scheme is eligible.
    let result = compressor.compress_child(
        &dict_child,
        &root_ctx,
        RestrictedAllowlistParent.id(),
        0,
        &mut ctx,
    )?;
    assert!(result.is::<Dict>(), "{}", result.encoding_id());

    // Canonical children still go through `compress_canonical` (e.g. constant collapse).
    let canonical_const = PrimitiveArray::from_iter([7i32; 64]).into_array();
    let collapsed = compressor.compress_child(
        &canonical_const,
        &root_ctx,
        RestrictedAllowlistParent.id(),
        0,
        &mut ctx,
    )?;
    assert!(collapsed.is::<Constant>(), "{}", collapsed.encoding_id());
    Ok(())
}

/// Hands back its whole input, validity included, as a general-purpose compressor would store
/// it.
#[derive(Debug)]
struct WholeLeafScheme;

impl Scheme for WholeLeafScheme {
    fn scheme_name(&self) -> &'static str {
        "test.whole_leaf"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
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

    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        Encoded::of_leaf(data.array().clone(), |leaf, _| Ok(leaf))
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

/// An encoding of a whole leaf receives the leaf with its validity bitmap compressed, here to a
/// constant.
#[test]
fn whole_leaf_encoding_stores_compressed_validity() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&WholeLeafScheme]);
    let input = PrimitiveArray::new(
        (0..4096).collect::<Buffer<i32>>(),
        Validity::Array(BoolArray::from_iter(iter::repeat_n(true, 4096)).into_array()),
    )
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&input, &mut ctx)?;
    assert_arrays_eq!(&compressed, &input, &mut ctx);

    let bitmap = compressed.slots()[PrimitiveSlots::VALIDITY_CHILD]
        .as_ref()
        .ok_or_else(|| vortex_err!("expected the validity bitmap to stay a slot"))?;
    assert!(bitmap.is::<Constant>(), "{}", bitmap.encoding_id());
    Ok(())
}

#[test]
fn test_container_children_preserve_is_sample() -> VortexResult<()> {
    let comp = CascadingCompressor::new(Vec::new());
    let mut ctx = SESSION.create_execution_ctx();
    let sample_ctx = CompressorContext::new().with_sampling();

    let const_i32 = PrimitiveArray::from_iter(iter::repeat_n(42i32, 1024)).into_array();
    let const_u64 = PrimitiveArray::from_iter(iter::repeat_n(7u64, 1024)).into_array();

    let fsl = FixedSizeListArray::new(const_u64, 16, Validity::NonNullable, 64).into_array();
    let ext_dtype = ExtDType::<BytesExtension>::try_new(0, Nullability::NonNullable)?.erased();
    let list = ListArray::try_new(
        const_i32.clone(),
        PrimitiveArray::from_iter(0..=1024u32).into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();
    let list_view = ListViewArray::try_new(
        const_i32.clone(),
        PrimitiveArray::new(
            (0..1024u32).map(|i| 1023 - i).collect::<Buffer<u32>>(),
            Validity::NonNullable,
        )
        .into_array(),
        PrimitiveArray::new(
            iter::repeat_n(1u32, 1024).collect::<Buffer<u32>>(),
            Validity::NonNullable,
        )
        .into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();
    let union = UnionArray::try_new(
        PrimitiveArray::from_iter(iter::repeat_n(0u8, 1024)).into_array(),
        UnionVariants::try_new(
            ["a"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            vec![0],
        )?,
        vec![const_i32.clone()],
    )?
    .into_array();
    let struct_arr = StructArray::try_new(
        FieldNames::from(["a"]),
        vec![const_i32.clone()],
        1024,
        Validity::NonNullable,
    )?
    .into_array();
    let ext = ExtensionArray::new(ext_dtype, fsl.clone()).into_array();
    let map = map_array_from_rows(&vec![Some(vec![(1, Some("same"))]); 128], true)?;
    let variant_core = ConstantArray::new(
        Scalar::variant(Scalar::primitive(1i32, Nullability::NonNullable)),
        1024,
    )
    .into_array();
    let variant = VariantArray::try_new(variant_core, Some(const_i32))?;

    for (name, input) in [
        ("Struct", struct_arr),
        ("Union", union),
        ("FixedSizeList", fsl.clone()),
        ("Extension", ext),
        ("List", list),
        ("ListView", list_view),
        ("Map", map),
    ] {
        let compressed = comp.compress_canonical(
            input.clone().execute::<Canonical>(&mut ctx)?,
            sample_ctx.clone(),
            &mut ctx,
        )?;
        assert!(
            !compressed
                .depth_first_traversal()
                .any(|a| a.is::<Constant>()),
            "sample {name} compression must not collapse constant children to Constant:\n{}",
            compressed.display_tree()
        );
        assert_arrays_eq!(&compressed, &input, &mut ctx);
    }

    let compressed_variant =
        comp.compress_canonical(Canonical::Variant(variant.clone()), sample_ctx, &mut ctx)?;
    let shredded = compressed_variant
        .as_::<Variant>()
        .shredded()
        .ok_or_else(|| vortex_err!("expected shredded child"))?;
    assert!(
        !shredded.depth_first_traversal().any(|a| a.is::<Constant>()),
        "sample Variant shredded child must not collapse to Constant:\n{}",
        shredded.display_tree()
    );
    assert_arrays_eq!(&compressed_variant, &variant.into_array(), &mut ctx);

    // Non-sample compression still collapses constant elements.
    let full_fsl = comp.compress(&fsl, &mut ctx)?;
    assert!(full_fsl.depth_first_traversal().any(|a| a.is::<Constant>()));
    Ok(())
}

/// Dictionary-encodes `[1, 2, 1, 2, ...]` as an encoding of its codes, wrapped in a dictionary
/// that registers its values.
#[derive(Debug)]
struct WrappingDictScheme;

impl Scheme for WrappingDictScheme {
    fn scheme_name(&self) -> &'static str {
        "test.wrapping_dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches_integer_primitive(canonical)
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

    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let codes = PrimitiveArray::new(
            [0u8, 1]
                .into_iter()
                .cycle()
                .take(data.array_len())
                .collect::<Buffer<u8>>(),
            Validity::NonNullable,
        )
        .into_array();
        let values = PrimitiveArray::new(buffer![1i32, 2], Validity::NonNullable).into_array();
        let codes = Encoded::new(
            vec![Child::new(self, 0, codes)],
            Vec::new(),
            |children, _| {
                let [codes] = children.compressed()?;
                Ok(codes)
            },
        );
        Ok(codes.wrap(
            vec![Child::new(self, 1, values)],
            Vec::new(),
            |codes, children, _| {
                let [values] = children.compressed()?;
                Ok(DictArray::try_new(codes, values)?.into_array())
            },
        ))
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

/// A wrapping encoding's constructor receives the inner encoding's output, and takes the children
/// registered after the inner encoding's own.
#[test]
fn wrapped_encoding_builds_from_inner_output() -> VortexResult<()> {
    let compressor = CascadingCompressor::new(vec![&WrappingDictScheme]);
    let input = alternating_ints();
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(&input, &mut ctx)?;
    assert!(compressed.is::<Dict>(), "{}", compressed.encoding_id());
    assert_arrays_eq!(&compressed, &input, &mut ctx);
    Ok(())
}
