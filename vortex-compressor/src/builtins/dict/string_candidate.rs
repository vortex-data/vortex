// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bounded trial compression for UTF-8 dictionary selection.
//!
//! Ordinary samples can miss repeated strings when a column has thousands of distinct values.
//! This module builds one bounded dictionary candidate from the complete input.
//!
//! A distributed full-value probe protects high-cardinality inputs from that full pass.
//! A high-cardinality probe rejects Dictionary. An inconclusive probe uses the existing sample
//! estimator.
//!
//! If the trial dictionary wins, the compression phase reuses its completed output.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Bool;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::builders::dict::DictConstraints;
use vortex_array::builders::dict::dict_encode_with_input_byte_limit;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

use super::StringDictScheme;
use super::string::compress_dictionary;
use crate::CascadingCompressor;
use crate::compressor::estimate_compression_ratio_with_sampling;
use crate::compressor::sample_slices;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
use crate::scheme::EstimateScore;
use crate::scheme::EstimateVerdict;
use crate::stats::ArrayAndStats;

/// Limits the number of entries so that trial codes remain two bytes wide.
const MAX_DICTIONARY_ENTRIES: usize = u16::MAX as usize + 1;

/// Limits the value and view storage in a trial dictionary.
const MAX_DICTIONARY_BYTES: usize = 1 << 22;

/// Limits code storage while the trial scans the complete input.
const MAX_DICTIONARY_CODE_BYTES: usize = 1 << 24;

/// Limits complete input bytes hashed while constructing a trial dictionary.
const MAX_DICTIONARY_INPUT_BYTES: usize = 1 << 28;

/// Limits complete value bytes hashed by the preliminary probe.
const MAX_PROBE_BYTES: usize = 1 << 20;

/// Samples this many adjacent rows from each probe range.
const PROBE_RANGE_SIZE: u32 = 64;

/// Distributes this many probe ranges across the input.
const PROBE_RANGE_COUNT: u32 = 128;

/// Sets the numerator for the 75% work-admission threshold.
const PROBE_DISTINCT_NUMERATOR: usize = 3;

/// Sets the denominator for the 75% work-admission threshold.
const PROBE_DISTINCT_DENOMINATOR: usize = 4;

/// Stores a completed dictionary result for reuse by the compression phase.
///
/// The containing [`ArrayAndStats`] exists for one selection and compression call. The cached
/// result therefore cannot cross compressor configurations or compression contexts.
#[derive(Debug)]
pub(super) struct CachedStringDictionary(ArrayRef);

impl CachedStringDictionary {
    /// Returns the completed dictionary result.
    pub(super) fn array(&self) -> &ArrayRef {
        &self.0
    }
}

/// Selects trial dictionary construction or ordinary sample estimation.
#[derive(Debug, PartialEq, Eq)]
enum ProbeResult {
    /// Permits bounded trial dictionary construction.
    Candidate,
    /// Rejects Dictionary for a high-cardinality input.
    Skip,
    /// Uses ordinary sample estimation because the probe did not finish.
    Inconclusive,
}

/// Reports the result of bounded trial dictionary construction.
enum CandidateResult {
    /// Contains a complete trial dictionary.
    Complete(DictArray),
    /// Reports that a preflight bound prevented trial construction.
    NotAttempted,
    /// Reports that trial construction exhausted a bound after work started.
    Exhausted,
}

/// Returns a deferred estimate that can reuse a completed dictionary result.
pub(super) fn string_dictionary_estimate() -> CompressionEstimate {
    CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
        estimate_string_dictionary,
    )))
}

/// Resolves the deferred dictionary estimate.
fn estimate_string_dictionary(
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    best_so_far: Option<EstimateScore>,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<EstimateVerdict> {
    match probe(data)? {
        ProbeResult::Candidate => {}
        ProbeResult::Skip => return Ok(EstimateVerdict::Skip),
        ProbeResult::Inconclusive => {
            return sample_verdict(compressor, data, best_so_far, compress_ctx, exec_ctx);
        }
    }

    let dictionary = match build_candidate(data, exec_ctx)? {
        CandidateResult::Complete(dictionary) => dictionary,
        CandidateResult::NotAttempted => {
            return sample_verdict(compressor, data, best_so_far, compress_ctx, exec_ctx);
        }
        CandidateResult::Exhausted => return Ok(EstimateVerdict::Skip),
    };
    let compressed = compress_dictionary(compressor, &dictionary, compress_ctx, exec_ctx)?;
    let score = EstimateScore::from_sample_sizes(data.array().nbytes(), compressed.nbytes());
    if !score_is_best(score, best_so_far) {
        return Ok(EstimateVerdict::Skip);
    }

    data.get_or_insert_with(|| CachedStringDictionary(compressed));
    Ok(score_verdict(score))
}

/// Runs the ordinary sample estimate and applies the current threshold.
fn sample_verdict(
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    best_so_far: Option<EstimateScore>,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<EstimateVerdict> {
    let score = estimate_compression_ratio_with_sampling(
        compressor,
        &StringDictScheme,
        data.array(),
        compress_ctx,
        exec_ctx,
    )?;
    if score_is_best(score, best_so_far) {
        Ok(score_verdict(score))
    } else {
        Ok(EstimateVerdict::Skip)
    }
}

/// Returns whether a score is valid and beats the current best score.
fn score_is_best(score: EstimateScore, best_so_far: Option<EstimateScore>) -> bool {
    score.is_valid() && best_so_far.is_none_or(|best| score.beats(best))
}

/// Converts a score into a terminal estimate.
fn score_verdict(score: EstimateScore) -> EstimateVerdict {
    match score {
        EstimateScore::FiniteCompression(ratio) => EstimateVerdict::Ratio(ratio),
        EstimateScore::ZeroBytes => EstimateVerdict::Skip,
    }
}

/// Checks complete values from distributed ranges before trial construction.
fn probe(data: &ArrayAndStats) -> VortexResult<ProbeResult> {
    let array = data.array_as_varbinview();
    let views = array.views();
    let validity = array.validity()?;
    let validity_bits = match &validity {
        vortex_array::validity::Validity::NonNullable
        | vortex_array::validity::Validity::AllValid => None,
        vortex_array::validity::Validity::AllInvalid => return Ok(ProbeResult::Inconclusive),
        vortex_array::validity::Validity::Array(validity) => {
            let Some(validity) = validity.as_opt::<Bool>() else {
                return Ok(ProbeResult::Inconclusive);
            };
            Some(validity.to_bit_buffer())
        }
    };
    let sample_ranges = sample_slices(array.len(), PROBE_RANGE_SIZE, PROBE_RANGE_COUNT);
    let sample_rows = sample_ranges
        .iter()
        .map(|(start, end)| end - start)
        .sum::<usize>();
    let mut distinct_values = HashSet::with_capacity(sample_rows);
    let mut probed_bytes = 0usize;
    let mut probed_rows = 0usize;
    let mut saw_null = false;

    let maximum_range_length = sample_ranges
        .iter()
        .map(|(start, end)| end - start)
        .max()
        .unwrap_or_default();

    'probe: for offset in 0..maximum_range_length {
        for &(start, end) in &sample_ranges {
            let index = start + offset;
            if index >= end {
                continue;
            }
            probed_rows += 1;
            if validity_bits
                .as_ref()
                .is_some_and(|validity| !validity.value(index))
            {
                saw_null = true;
                continue;
            }
            let value = view_bytes(&array, &views[index]);
            let Some(next_probed_bytes) = probed_bytes.checked_add(value.len()) else {
                break 'probe;
            };
            if next_probed_bytes > MAX_PROBE_BYTES {
                break 'probe;
            }
            probed_bytes = next_probed_bytes;
            distinct_values.insert(value);
        }
    }

    if probed_rows == 0 {
        return Ok(ProbeResult::Inconclusive);
    }
    if probed_rows != sample_rows {
        return Ok(ProbeResult::Inconclusive);
    }

    let distinct_values = distinct_values.len() + usize::from(saw_null);
    if distinct_values * PROBE_DISTINCT_DENOMINATOR <= probed_rows * PROBE_DISTINCT_NUMERATOR {
        Ok(ProbeResult::Candidate)
    } else {
        Ok(ProbeResult::Skip)
    }
}

/// Builds a dictionary within explicit storage and work bounds.
fn build_candidate(
    data: &ArrayAndStats,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<CandidateResult> {
    let Some(code_bytes) = candidate_code_bytes(data.array_len(), MAX_DICTIONARY_ENTRIES) else {
        return Ok(CandidateResult::NotAttempted);
    };
    if code_bytes > MAX_DICTIONARY_CODE_BYTES {
        return Ok(CandidateResult::NotAttempted);
    }

    let constraints = DictConstraints {
        max_bytes: MAX_DICTIONARY_BYTES,
        max_len: MAX_DICTIONARY_ENTRIES,
    };
    let dictionary = dict_encode_with_input_byte_limit(
        data.array(),
        &constraints,
        MAX_DICTIONARY_INPUT_BYTES,
        exec_ctx,
    )?;
    if dictionary.len() != data.array_len() {
        return Ok(CandidateResult::Exhausted);
    }

    Ok(CandidateResult::Complete(dictionary))
}

/// Returns the code storage required by the constrained dictionary builder.
fn candidate_code_bytes(row_count: usize, maximum_dictionary_values: usize) -> Option<usize> {
    let code_width = if maximum_dictionary_values <= usize::from(u8::MAX) + 1 {
        size_of::<u8>()
    } else if maximum_dictionary_values <= usize::from(u16::MAX) + 1 {
        size_of::<u16>()
    } else {
        size_of::<u32>()
    };
    row_count.checked_mul(code_width)
}

/// Returns the complete value referenced by a binary view.
fn view_bytes<'a>(array: &'a ArrayView<'a, VarBinView>, view: &'a BinaryView) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let reference = view.as_view();
        &array.buffer(reference.buffer_index as usize)[reference.as_range()]
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;

    use super::CachedStringDictionary;
    use super::CandidateResult;
    use super::PROBE_RANGE_COUNT;
    use super::PROBE_RANGE_SIZE;
    use super::ProbeResult;
    use super::build_candidate;
    use super::probe;
    use super::sample_slices;
    use super::string_dictionary_estimate;
    use crate::CascadingCompressor;
    use crate::builtins::StringDictScheme;
    use crate::scheme::CompressionEstimate;
    use crate::scheme::CompressorContext;
    use crate::scheme::DeferredEstimate;
    use crate::scheme::EstimateVerdict;
    use crate::scheme::Scheme;
    use crate::stats::ArrayAndStats;

    /// Builds compression data for string candidate tests.
    fn string_data(values: &[Option<String>]) -> ArrayAndStats {
        let nullability = if values.iter().any(Option::is_none) {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        };
        let array = VarBinViewArray::from_iter(
            values.iter().map(|value| value.as_deref()),
            DType::Utf8(nullability),
        )
        .into_array();
        ArrayAndStats::new(array, Default::default())
    }

    /// Resolves the Dictionary estimate without another candidate scheme.
    fn estimate_dictionary(
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        exec_ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<EstimateVerdict> {
        let callback = match string_dictionary_estimate() {
            CompressionEstimate::Deferred(DeferredEstimate::Callback(callback)) => callback,
            estimate => {
                return Err(vortex_error::vortex_err!(
                    "dictionary estimation returned {estimate:?}"
                ));
            }
        };
        callback(compressor, data, None, CompressorContext::new(), exec_ctx)
    }

    #[test]
    fn probe_distinct_ratio_boundary() -> VortexResult<()> {
        for (distinct_values, expected) in
            [(6144, ProbeResult::Candidate), (6145, ProbeResult::Skip)]
        {
            let values = (0..8192)
                .map(|index| {
                    Some(format!(
                        "common-prefix-value-{:08x}",
                        index % distinct_values
                    ))
                })
                .collect::<Vec<_>>();
            let data = string_data(&values);

            assert_eq!(probe(&data)?, expected);
        }
        Ok(())
    }

    #[test]
    fn probe_reports_inconclusive_at_byte_limit() -> VortexResult<()> {
        let mut values = (0..65_536)
            .map(|index| Some(format!("{index:0256x}")))
            .collect::<Vec<_>>();
        for (start, end) in sample_slices(values.len(), PROBE_RANGE_SIZE, PROBE_RANGE_COUNT)
            .into_iter()
            .take(16)
        {
            values[start..end].fill(Some("x".repeat(256)));
        }
        let data = string_data(&values);
        assert_eq!(probe(&data)?, ProbeResult::Inconclusive);
        Ok(())
    }

    #[test]
    fn probe_counts_null_as_one_repeated_value() -> VortexResult<()> {
        let values = (0..8192)
            .map(|index| (index % 10 == 0).then(|| format!("unique-non-null-value-{index:08x}")))
            .collect::<Vec<_>>();
        let data = string_data(&values);

        assert_eq!(probe(&data)?, ProbeResult::Candidate);
        Ok(())
    }

    #[test]
    fn high_cardinality_probe_is_terminal() -> VortexResult<()> {
        let values = (0..8192)
            .map(|index| Some(format!("unique-value-{index:08x}")))
            .collect::<Vec<_>>();
        let data = string_data(&values);
        let compressor = CascadingCompressor::new(vec![&StringDictScheme]);
        let mut exec_ctx = vortex_array::array_session().create_execution_ctx();

        assert_eq!(probe(&data)?, ProbeResult::Skip);
        assert!(matches!(
            estimate_dictionary(&compressor, &data, &mut exec_ctx)?,
            EstimateVerdict::Skip
        ));
        Ok(())
    }

    #[test]
    fn candidate_stops_when_dictionary_storage_exceeds_limit() -> VortexResult<()> {
        let suffix = "x".repeat(70_000);
        let distinct_values = (0..64)
            .map(|value| format!("{value:08x}-{suffix}"))
            .collect::<Vec<_>>();
        let values = (0..128)
            .map(|index| Some(distinct_values[index % distinct_values.len()].clone()))
            .collect::<Vec<_>>();
        let data = string_data(&values);
        let mut exec_ctx = vortex_array::array_session().create_execution_ctx();

        assert!(matches!(
            build_candidate(&data, &mut exec_ctx)?,
            CandidateResult::Exhausted
        ));
        Ok(())
    }

    #[test]
    fn candidate_uses_full_u16_code_range() -> VortexResult<()> {
        let values = (0..=u16::MAX)
            .map(|value| Some(format!("{value:08x}")))
            .collect::<Vec<_>>();
        let data = string_data(&values);
        let mut exec_ctx = vortex_array::array_session().create_execution_ctx();

        assert!(matches!(
            build_candidate(&data, &mut exec_ctx)?,
            CandidateResult::Complete(_)
        ));

        let mut values_past_limit = values;
        values_past_limit.push(Some("past-u16-code-range".to_owned()));
        let data = string_data(&values_past_limit);
        assert!(matches!(
            build_candidate(&data, &mut exec_ctx)?,
            CandidateResult::Exhausted
        ));
        Ok(())
    }

    #[test]
    fn completed_candidate_is_reused() -> VortexResult<()> {
        let values = (0..65_536)
            .map(|index| Some(format!("common-prefix-value-{:08x}", index % 4096)))
            .collect::<Vec<_>>();
        let data = string_data(&values);
        let compressor = CascadingCompressor::new(vec![&StringDictScheme]);
        let mut exec_ctx = vortex_array::array_session().create_execution_ctx();
        let verdict = estimate_dictionary(&compressor, &data, &mut exec_ctx)?;
        assert!(matches!(verdict, EstimateVerdict::Ratio(_)));
        let cached = data
            .get::<CachedStringDictionary>()
            .ok_or_else(|| vortex_error::vortex_err!("completed candidate was not cached"))?;
        let compressed = StringDictScheme.compress(
            &compressor,
            &data,
            CompressorContext::new(),
            &mut exec_ctx,
        )?;

        assert!(ArrayRef::ptr_eq(cached.array(), &compressed));
        Ok(())
    }
}
