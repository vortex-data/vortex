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
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::builders::dict::dict_encode;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

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

/// Maximum average string length for the aggressive near-unique veto.
///
/// Short structured strings such as `t_time_id`, `d_date_id`, and ISO-like timestamp text can
/// share a tiny set of `(length, prefix)` signatures while still being almost fully unique. Those
/// columns are exactly where the permissive prefix-only dict gate misfires: dictionary encoding
/// sees the shared structure, but the extra codes layer adds overhead that direct FSST avoids.
///
/// We therefore let the sample preflight reject dictionary encoding outright when the sampled
/// strings are both short and mostly unique. The short-string cutoff is intentionally conservative:
/// URL and file-path columns are much longer, so they do not hit this veto.
const STRING_DICT_NEAR_UNIQUE_SHORT_AVG_LEN_MAX: f64 = 32.0;

/// Distinct-density threshold for the aggressive short-string veto.
///
/// Once roughly three quarters of sampled short strings are exact-value distinct, dict is rarely a
/// good trade. The values child stays large, and the codes layer becomes almost pure overhead.
const STRING_DICT_NEAR_UNIQUE_SHORT_DISTINCT_DENSITY: f64 = 0.75;

/// Maximum average string length for the softer near-unique veto.
///
/// Medium-length strings can still be bad dict candidates when they are almost entirely unique,
/// but we keep this band separate from the short-string rule so that long URL/path-like values are
/// still decided by the normal raw-dictionary upper bound and sampled comparison.
const STRING_DICT_NEAR_UNIQUE_MEDIUM_AVG_LEN_MAX: f64 = 64.0;

/// Distinct-density threshold for the softer medium-length veto.
///
/// This band only rejects samples that are extremely close to fully unique. Anything less clear
/// still falls through to the existing preflight and, if needed, normal sampled evaluation.
const STRING_DICT_NEAR_UNIQUE_MEDIUM_DISTINCT_DENSITY: f64 = 0.9;

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
/// 2. Compute the sample's exact full-value distinct density and average length.
/// 3. Immediately reject dictionary encoding for short or medium-length samples that are nearly
///    fully unique. This catches structured IDs and date strings whose coarse prefix stats look
///    dictionary-friendly even though their exact values are not.
/// 4. Dictionary-encode the sample without recursively compressing the children.
/// 5. Treat the resulting raw sample ratio as a floor, then inflate it by
///    [`STRING_DICT_RAW_SAMPLE_RATIO_UPLIFT`] to get a generous "best plausible" ratio.
/// 6. If that optimistic ratio still cannot beat the current best scheme, skip dictionary
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
    let sample_utf8 = sample_array
        .as_opt::<VarBinView>()
        .vortex_expect("string dict preflight only runs on canonical UTF-8 samples")
        .into_owned();

    if should_skip_string_dict_for_near_unique_sample(&sample_utf8)? {
        return Ok(SamplePreflightVerdict::Skip);
    }

    let threshold = best_so_far.and_then(EstimateScore::finite_ratio);

    if let Some(threshold) = threshold.filter(|&t| t >= STRING_DICT_PREFLIGHT_MIN_THRESHOLD) {
        let max_ratio = raw_string_dict_ratio_upper_bound(&sample_array)?;
        if max_ratio <= threshold {
            return Ok(SamplePreflightVerdict::Skip);
        }
    }

    Ok(SamplePreflightVerdict::Sample)
}

/// Exact sample-level distinctness stats used only by the string dict preflight.
struct SampleStringValueStats {
    /// Number of exact distinct non-null sample values.
    exact_distinct_count: u32,
    /// Total byte length of all non-null sample values.
    total_value_bytes: u64,
    /// Number of non-null sample values.
    value_count: u32,
}

/// Returns whether sampled strings are too unique for dictionary encoding to be competitive.
///
/// The prefix-only dict gate is intentionally permissive so that repeated long URLs and file paths
/// stay eligible. That same permissiveness admits structured strings that are almost entirely
/// unique, such as surrogate IDs and timestamp text. This helper uses the sampled strings'
/// full-byte identity, not the coarse `(length, prefix)` sketch, to veto those near-unique cases
/// before dictionary sampling pays for raw dict construction or recursive child compression.
fn should_skip_string_dict_for_near_unique_sample(
    sample_utf8: &VarBinViewArray,
) -> VortexResult<bool> {
    let sample_stats = sampled_string_value_stats(sample_utf8)?;

    if sample_stats.value_count == 0 {
        return Ok(false);
    }

    let value_count = f64::from(sample_stats.value_count);
    let avg_value_len = sample_stats.total_value_bytes as f64 / value_count;
    let exact_distinct_density = f64::from(sample_stats.exact_distinct_count) / value_count;

    Ok((avg_value_len <= STRING_DICT_NEAR_UNIQUE_SHORT_AVG_LEN_MAX
        && exact_distinct_density >= STRING_DICT_NEAR_UNIQUE_SHORT_DISTINCT_DENSITY)
        || (avg_value_len <= STRING_DICT_NEAR_UNIQUE_MEDIUM_AVG_LEN_MAX
            && exact_distinct_density >= STRING_DICT_NEAR_UNIQUE_MEDIUM_DISTINCT_DENSITY))
}

/// Computes exact distinctness stats for a sampled UTF-8 array.
///
/// This is intentionally narrower than global `StringStats`: it only runs inside the dict
/// preflight after the scheme has already cleared the cheap prefix gate, and it hashes the full
/// sampled strings rather than a sketch. Borrowed byte slices are inserted directly into the hash
/// set so we can count exact sample distinctness without copying string payloads.
fn sampled_string_value_stats(
    sample_utf8: &VarBinViewArray,
) -> VortexResult<SampleStringValueStats> {
    sample_utf8.with_iterator(|iter| {
        let mut distinct_values = HashSet::<&[u8]>::with_capacity(sample_utf8.len() / 2);
        let mut value_count = 0usize;
        let mut total_value_bytes = 0u64;

        for value in iter.flatten() {
            distinct_values.insert(value);
            value_count += 1;
            total_value_bytes +=
                u64::try_from(value.len()).vortex_expect("usize string lengths must fit in u64");
        }

        Ok(SampleStringValueStats {
            exact_distinct_count: u32::try_from(distinct_values.len())?,
            total_value_bytes,
            value_count: u32::try_from(value_count)?,
        })
    })
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

    #[test]
    fn string_dict_preflight_skips_near_unique_short_structured_strings() -> VortexResult<()> {
        let strings = VarBinViewArray::from_iter(
            (0..4096).map(|idx| Some(format!("TIME-{idx:08}"))),
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

        let verdict = preflight(&data, None, CompressorContext::new(), &mut ctx)?;

        assert!(matches!(verdict, SamplePreflightVerdict::Skip));
        Ok(())
    }

    #[test]
    fn string_dict_preflight_keeps_repeated_long_paths_in_play() -> VortexResult<()> {
        let values = [
            "https://example.com/articles/releases/2025/01/0001/index.html",
            "https://example.com/articles/releases/2025/01/0002/index.html",
            "https://example.com/articles/releases/2025/01/0003/index.html",
            "https://example.com/articles/releases/2025/01/0004/index.html",
        ];
        let strings = VarBinViewArray::from_iter(
            (0..4096).map(|idx| Some(values[idx % values.len()])),
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

        let verdict = preflight(&data, None, CompressorContext::new(), &mut ctx)?;

        assert!(matches!(verdict, SamplePreflightVerdict::Sample));
        Ok(())
    }
}
