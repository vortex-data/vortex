// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UTF8-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted
//! for external compatibility.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::builders::dict::dict_encode;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::CascadingCompressor;
use crate::builtins::IntDictScheme;
use crate::builtins::StringDictScheme;
use crate::builtins::is_utf8_string;
use crate::ctx::CompressorContext;
use crate::estimate::CompressionEstimate;
use crate::estimate::DeferredEstimate;
use crate::estimate::EstimateScore;
use crate::estimate::EstimateVerdict;
use crate::estimate::SamplePreflightVerdict;
use crate::sample::SAMPLE_SIZE;
use crate::sample::sample;
use crate::sample::sample_count_approx_one_percent;
use crate::scheme::ChildSelection;
use crate::scheme::DescendantExclusion;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;

/// Minimum incumbent ratio before the string dict preflight tries to prune itself.
///
/// When the current best ratio is still close to `1.0`, the preflight does not try to be clever:
/// it lets normal sampling run because small threshold differences are exactly where rough
/// upper-bound models are least trustworthy. The preflight only activates once another scheme
/// already has a clearly worthwhile lead.
const STRING_DICT_PREFLIGHT_MIN_THRESHOLD: f64 = 2.0;

/// Optimistic improvement factor applied to the raw dictionary-sample ratio.
///
/// The raw `dict_encode(sample)` layout does not include recursive compression of the dictionary
/// values child or the codes child, so it is a pessimistic estimate of the final dictionary
/// result. The callback multiplies that raw ratio by this factor to get an intentionally generous
/// upper bound before deciding to skip sampled compression.
const STRING_DICT_RAW_SAMPLE_RATIO_UPLIFT: f64 = 3.0;

impl Scheme for StringDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_utf8_string(canonical)
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        GenerateStatsOptions {
            count_distinct_values: true,
        }
    }

    /// Children: values=0, codes=1.
    fn num_children(&self) -> usize {
        2
    }

    /// String dict codes (child 1) are compact unsigned integers that should not be dict-encoded
    /// again.
    ///
    /// Additional exclusions for codes (IntSequenceScheme, FoRScheme, ZigZagScheme, SparseScheme,
    /// RunEndScheme, RLE, etc.) are expressed as pull rules on those schemes in `vortex-btrblocks`.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::One(1),
        }]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let stats = data.string_stats(exec_ctx);

        if stats.value_count() == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // This gate is intentionally permissive. Dictionary encoding only needs a cheap signal that
        // repeated categories are plausible; sampling decides whether the final dictionary layout
        // is actually better than the alternatives. Using the suffix-aware string distinct count
        // here turned out to be too strict for URL- and path-like columns: those arrays often have
        // high full-value cardinality while still benefiting from dictionary encoding because many
        // rows reuse exact values across the larger compression batch. The coarser
        // `(length, first four bytes)` count keeps those candidates eligible for sampling instead
        // of skipping dictionary encoding up front.
        let estimated_distinct_values_count =
            stats.estimated_prefix_distinct_count().vortex_expect(
                "this must be present since `DictScheme` declared that we need distinct values",
            );

        // If > 50% of the values are distinct, skip dictionary scheme.
        if estimated_distinct_values_count > stats.value_count() / 2 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::PreflightThenSample(Box::new(
            |data, best_so_far, compress_ctx, exec_ctx| {
                string_dict_sample_preflight(data, best_so_far, compress_ctx, exec_ctx)
            },
        )))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let dict = dict_encode(data.array())?;

        // Values = child 0.
        let compressed_values =
            compressor.compress_child(dict.values(), &compress_ctx, self.id(), 0, exec_ctx)?;

        // Codes = child 1.
        let narrowed_codes = dict
            .codes()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow()?
            .into_array();
        let compressed_codes =
            compressor.compress_child(&narrowed_codes, &compress_ctx, self.id(), 1, exec_ctx)?;

        // SAFETY: compressing codes or values does not alter the invariants.
        unsafe {
            Ok(
                DictArray::new_unchecked(compressed_codes, compressed_values)
                    .set_all_values_referenced(dict.has_all_values_referenced())
                    .into_array(),
            )
        }
    }
}

/// Decides whether string dictionary encoding should proceed to normal sampled evaluation.
///
/// The string dictionary gate is intentionally permissive, which means many columns arrive here as
/// "maybe" candidates. Fully sampling dictionary compression can be expensive because it builds the
/// sample dictionary and then recursively compresses both children. This preflight therefore runs a
/// cheap upper-bound check first:
///
/// 1. Reuse the same sample that normal ratio estimation would use.
/// 2. Dictionary-encode that sample without recursively compressing the children.
/// 3. Treat the resulting raw sample ratio as a floor, then inflate it by
///    [`STRING_DICT_RAW_SAMPLE_RATIO_UPLIFT`] to get a generous "best plausible" ratio.
/// 4. If that optimistic ratio still cannot beat the current best scheme, skip dictionary
///    sampling entirely. Otherwise, let the compressor run the normal sampled compression path.
///
/// This keeps the actual sample compression logic centralized in the compressor instead of
/// reimplementing it inside the scheme.
fn string_dict_sample_preflight(
    data: &ArrayAndStats,
    best_so_far: Option<EstimateScore>,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<SamplePreflightVerdict> {
    let sample_array = estimation_sample_array(data.array(), &compress_ctx, exec_ctx)?;
    let threshold = best_so_far.and_then(EstimateScore::finite_ratio);

    if let Some(threshold) = threshold.filter(|&t| t >= STRING_DICT_PREFLIGHT_MIN_THRESHOLD) {
        let max_ratio = raw_string_dict_ratio_upper_bound(&sample_array)?;
        if max_ratio <= threshold {
            return Ok(SamplePreflightVerdict::Skip);
        }
    }

    Ok(SamplePreflightVerdict::Sample)
}

/// Returns the exact sample array the compressor would use for sampled ratio estimation.
fn estimation_sample_array(
    array: &ArrayRef,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if compress_ctx.is_sample() {
        return Ok(array.clone());
    }

    let sample_count = sample_count_approx_one_percent(array.len());
    let canonical: Canonical = sample(array, SAMPLE_SIZE, sample_count).execute(exec_ctx)?;
    Ok(canonical.into_array())
}

/// Returns an optimistic upper bound for the final string dictionary ratio on `sample_array`.
///
/// This uses the raw `dict_encode` layout as a cheap floor for dictionary compression and then
/// inflates that ratio to account for recursive child compression. The result is intentionally
/// generous because it is only used to prove that dictionary cannot plausibly beat the current
/// best scheme.
fn raw_string_dict_ratio_upper_bound(sample_array: &ArrayRef) -> VortexResult<f64> {
    let raw_dict_nbytes = dict_encode(sample_array)?.into_array().nbytes().max(1) as f64;
    let raw_ratio = sample_array.nbytes() as f64 / raw_dict_nbytes;
    Ok(raw_ratio * STRING_DICT_RAW_SAMPLE_RATIO_UPLIFT)
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn string_dict_stays_eligible_for_common_prefix_varying_tail_values() {
        let strings = VarBinViewArray::from_iter(
            [
                Some("https://example.com/events/0000"),
                Some("https://example.com/events/0001"),
                Some("https://example.com/events/0002"),
                Some("https://example.com/events/0003"),
            ],
            DType::Utf8(Nullability::NonNullable),
        );
        let data = ArrayAndStats::new(
            strings.into_array(),
            GenerateStatsOptions {
                count_distinct_values: true,
            },
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        assert!(matches!(
            StringDictScheme.expected_compression_ratio(&data, CompressorContext::new(), &mut ctx,),
            CompressionEstimate::Deferred(DeferredEstimate::PreflightThenSample(_))
        ));
    }

    #[test]
    fn string_dict_skips_when_prefix_cardinality_is_high() {
        let strings = VarBinViewArray::from_iter(
            [Some("aaaa"), Some("bbbb"), Some("cccc"), Some("dddd")],
            DType::Utf8(Nullability::NonNullable),
        );
        let data = ArrayAndStats::new(
            strings.into_array(),
            GenerateStatsOptions {
                count_distinct_values: true,
            },
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        assert!(matches!(
            StringDictScheme.expected_compression_ratio(&data, CompressorContext::new(), &mut ctx,),
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        ));
    }

    #[test]
    fn string_dict_preflight_skips_when_threshold_is_already_far_better() -> VortexResult<()> {
        let strings = VarBinViewArray::from_iter(
            [
                Some("https://example.com/events/0000"),
                Some("https://example.com/events/0001"),
                Some("https://example.com/events/0002"),
                Some("https://example.com/events/0003"),
            ],
            DType::Utf8(Nullability::NonNullable),
        );
        let data = ArrayAndStats::new(
            strings.into_array(),
            GenerateStatsOptions {
                count_distinct_values: true,
            },
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let estimate =
            StringDictScheme.expected_compression_ratio(&data, CompressorContext::new(), &mut ctx);
        let CompressionEstimate::Deferred(DeferredEstimate::PreflightThenSample(preflight)) =
            estimate
        else {
            panic!("expected deferred preflight");
        };

        let verdict = preflight(
            &data,
            Some(EstimateScore::FiniteCompression(100.0)),
            CompressorContext::new(),
            &mut ctx,
        )?;

        assert!(matches!(verdict, SamplePreflightVerdict::Skip));
        Ok(())
    }
}
