// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bounded trial compression for variable-width dictionary selection.
//!
//! Ordinary samples can miss repeated values when a column has thousands of distinct values, so
//! a sample-based estimate under-rates dictionaries for those columns. This module instead
//! builds one bounded dictionary candidate from the complete input.
//!
//! Before paying for that trial, a distributed probe of complete values estimates the number of
//! distinct values in the whole input. A cost model turns that estimate into an optimistic
//! compression ratio, and the trial runs only when that ratio could beat the best estimate so
//! far. A probe that cannot finish within its budget, and a dictionary too large for the trial,
//! fall back to the ordinary sample estimator.
//!
//! If the trial dictionary wins, the compression phase reuses its completed output.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::builders::dict::DictConstraints;
use vortex_array::builders::dict::dict_encode;
use vortex_array::builders::dict::dict_encode_with_constraints;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

use crate::CascadingCompressor;
use crate::compressor::estimate_compression_ratio_with_sampling;
use crate::compressor::sample_slices;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
use crate::scheme::EstimateScore;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
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

/// Stores a completed dictionary result for reuse by the compression phase.
///
/// The containing [`ArrayAndStats`] exists for one selection and compression call. The cached
/// result therefore cannot cross compressor configurations or compression contexts.
#[derive(Debug)]
pub(super) struct CachedBytesDictionary(ArrayRef);

impl CachedBytesDictionary {
    /// Returns the completed dictionary result.
    pub(super) fn array(&self) -> &ArrayRef {
        &self.0
    }
}

/// Returns a deferred estimate that can reuse a completed dictionary result.
pub(super) fn bytes_dictionary_estimate(scheme: &'static dyn Scheme) -> CompressionEstimate {
    CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
        move |compressor, data, best_so_far, compress_ctx, exec_ctx| {
            estimate_bytes_dictionary(
                scheme,
                compressor,
                data,
                best_so_far,
                compress_ctx,
                exec_ctx,
            )
        },
    )))
}

/// Dictionary-encodes the input, reusing the trial result when the estimate cached one.
pub(super) fn compress_bytes_dictionary(
    scheme: &'static dyn Scheme,
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(candidate) = data.get::<CachedBytesDictionary>() {
        return Ok(candidate.array().clone());
    }

    let dict = dict_encode(data.array(), exec_ctx)?;
    compress_dictionary(scheme, compressor, &dict, compress_ctx, exec_ctx)
}

/// Compresses the value and code children of a dictionary.
fn compress_dictionary(
    scheme: &'static dyn Scheme,
    compressor: &CascadingCompressor,
    dict: &DictArray,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Values = child 0.
    let compressed_values =
        compressor.compress_child(dict.values(), &compress_ctx, scheme.id(), 0, exec_ctx)?;

    // Codes = child 1.
    let narrowed_codes = dict
        .codes()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?
        .narrow(exec_ctx)?
        .into_array();
    let compressed_codes =
        compressor.compress_child(&narrowed_codes, &compress_ctx, scheme.id(), 1, exec_ctx)?;

    // SAFETY: compressing codes or values does not alter the invariants.
    unsafe {
        Ok(
            DictArray::new_unchecked(compressed_codes, compressed_values)
                .set_all_values_referenced(dict.has_all_values_referenced())
                .into_array(),
        )
    }
}

/// Resolves the deferred dictionary estimate.
fn estimate_bytes_dictionary(
    scheme: &'static dyn Scheme,
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    best_so_far: Option<EstimateScore>,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<EstimateVerdict> {
    let array = data.array_as_varbinview();
    let profile = InputProfile::new(&array, exec_ctx)?;
    if profile.valid_rows == 0 {
        return Ok(EstimateVerdict::Skip);
    }

    let Some(sample) = ProbeSample::new(&array, &profile.validity) else {
        return sample_verdict(
            scheme,
            compressor,
            data,
            best_so_far,
            compress_ctx,
            exec_ctx,
        );
    };
    match admit_trial(&profile, &sample, best_so_far) {
        Admission::Trial => {}
        Admission::Reject => return Ok(EstimateVerdict::Skip),
        Admission::TooLarge => {
            return sample_verdict(
                scheme,
                compressor,
                data,
                best_so_far,
                compress_ctx,
                exec_ctx,
            );
        }
    }

    let Some(dictionary) = build_candidate(data.array(), exec_ctx)? else {
        return sample_verdict(
            scheme,
            compressor,
            data,
            best_so_far,
            compress_ctx,
            exec_ctx,
        );
    };
    let compressed = compress_dictionary(scheme, compressor, &dictionary, compress_ctx, exec_ctx)?;
    let score = EstimateScore::from_sample_sizes(profile.logical_nbytes, compressed.nbytes());
    if !score_is_best(score, best_so_far) {
        return Ok(EstimateVerdict::Skip);
    }

    data.get_or_insert_with(|| CachedBytesDictionary(compressed));
    Ok(score_verdict(score))
}

/// Runs the ordinary sample estimate and applies the current threshold.
fn sample_verdict(
    scheme: &'static dyn Scheme,
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    best_so_far: Option<EstimateScore>,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<EstimateVerdict> {
    let score = estimate_compression_ratio_with_sampling(
        compressor,
        scheme,
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

/// Describes the complete input with one pass over its views.
struct InputProfile {
    /// Counts all rows, including nulls.
    rows: usize,
    /// Counts non-null rows.
    valid_rows: usize,
    /// Sums the lengths of non-null values.
    value_bytes: usize,
    /// Measures the input as if its data buffers held only referenced, non-null values.
    ///
    /// Sample estimates measure compacted samples, so the trial measures the same way.
    logical_nbytes: u64,
    /// Marks non-null rows.
    validity: Mask,
}

impl InputProfile {
    /// Profiles the complete input.
    fn new(array: &ArrayView<'_, VarBinView>, exec_ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let validity = array.validity()?.execute_mask(array.len(), exec_ctx)?;
        let views = array.views();
        let mut value_bytes = 0usize;
        let mut outlined_bytes = 0usize;
        let mut add = |view: &BinaryView| {
            let len = view.len() as usize;
            value_bytes += len;
            if !view.is_inlined() {
                outlined_bytes += len;
            }
        };
        match validity.bit_buffer() {
            AllOr::All => views.iter().for_each(&mut add),
            AllOr::None => {}
            AllOr::Some(bits) => bits.set_indices().for_each(|index| add(&views[index])),
        }

        let buffer_bytes = array
            .data_buffers()
            .iter()
            .map(|buffer| buffer.len() as u64)
            .sum::<u64>();
        let logical_nbytes = array.as_ref().nbytes() - buffer_bytes + outlined_bytes as u64;

        Ok(Self {
            rows: array.len(),
            valid_rows: validity.true_count(),
            value_bytes,
            logical_nbytes,
            validity,
        })
    }

    /// Returns the mean length of a non-null value.
    fn mean_value_bytes(&self) -> f64 {
        self.value_bytes as f64 / self.valid_rows as f64
    }
}

/// Summarizes complete values from distributed ranges of the input.
#[derive(Debug, PartialEq)]
struct ProbeSample {
    /// Counts probed non-null rows.
    rows: usize,
    /// Counts distinct probed non-null values.
    distinct: usize,
    /// Counts distinct values that occur exactly once in the probe.
    singletons: usize,
    /// Counts distinct values that occur exactly twice in the probe.
    doubletons: usize,
    /// Counts adjacent row pairs within a probe range.
    adjacent_pairs: usize,
    /// Counts adjacent row pairs within a probe range that hold different values.
    adjacent_changes: usize,
    /// Reports whether the probe covered every row of the input.
    complete: bool,
}

impl ProbeSample {
    /// Probes the input, or returns `None` when the probed values exceed the probe budget.
    fn new(array: &ArrayView<'_, VarBinView>, validity: &Mask) -> Option<Self> {
        let views = array.views();
        let ranges = sample_slices(array.len(), PROBE_RANGE_SIZE, PROBE_RANGE_COUNT);
        let complete = matches!(ranges.as_slice(), [(0, end)] if *end == array.len());

        let probe_bytes = ranges
            .iter()
            .flat_map(|&(start, end)| start..end)
            .filter(|&index| validity.value(index))
            .map(|index| views[index].len() as usize)
            .sum::<usize>();
        if probe_bytes > MAX_PROBE_BYTES {
            return None;
        }

        let mut counts = HashMap::<&[u8], u32>::new();
        let mut adjacent_pairs = 0;
        let mut adjacent_changes = 0;
        for (start, end) in ranges {
            let mut previous = None;
            for index in start..end {
                let value = validity
                    .value(index)
                    .then(|| view_bytes(array, &views[index]));
                if let Some(previous) = previous {
                    adjacent_pairs += 1;
                    adjacent_changes += usize::from(previous != value);
                }
                previous = Some(value);
                if let Some(value) = value {
                    *counts.entry(value).or_default() += 1;
                }
            }
        }

        let rows = counts.values().map(|&count| count as usize).sum::<usize>();
        if rows == 0 {
            return None;
        }
        Some(Self {
            rows,
            distinct: counts.len(),
            singletons: counts.values().filter(|&&count| count == 1).count(),
            doubletons: counts.values().filter(|&&count| count == 2).count(),
            adjacent_pairs,
            adjacent_changes,
            complete,
        })
    }

    /// Estimates the number of distinct non-null values in the complete input.
    ///
    /// Uses the Chao1 estimator, which tends to under-count. An under-count only costs a trial
    /// that the complete dictionary then rejects, while an over-count would skip a dictionary
    /// that wins.
    fn estimated_distinct(&self, valid_rows: usize) -> f64 {
        let distinct = self.distinct as f64;
        if self.complete {
            return distinct;
        }
        let singletons = self.singletons as f64;
        let doubletons = self.doubletons as f64;
        let unseen = if self.doubletons > 0 {
            singletons * singletons / (2.0 * doubletons)
        } else {
            singletons * (singletons - 1.0) / 2.0
        };
        (distinct + unseen).min(valid_rows as f64)
    }

    /// Estimates the fraction of rows that start a new run of equal values.
    fn run_start_fraction(&self) -> f64 {
        if self.adjacent_pairs == 0 {
            return 1.0;
        }
        (self.adjacent_changes as f64 / self.adjacent_pairs as f64)
            .max(1.0 / f64::from(PROBE_RANGE_SIZE))
    }
}

/// Decides whether to build a trial dictionary.
#[derive(Debug, PartialEq, Eq)]
enum Admission {
    /// Builds the trial dictionary.
    Trial,
    /// Rejects Dictionary because it cannot beat the best estimate so far.
    Reject,
    /// Leaves the decision to sampling because the trial would exceed its bounds.
    ///
    /// A dictionary larger than the trial allows can still win with wider codes.
    TooLarge,
}

/// Applies the trial bounds and the cost model.
fn admit_trial(
    profile: &InputProfile,
    sample: &ProbeSample,
    best_so_far: Option<EstimateScore>,
) -> Admission {
    let best_ratio = match best_so_far {
        Some(EstimateScore::FiniteCompression(ratio)) => ratio.max(1.0),
        Some(EstimateScore::ZeroBytes) => return Admission::Reject,
        None => 1.0,
    };
    let distinct = sample.estimated_distinct(profile.valid_rows);
    let has_nulls = profile.valid_rows < profile.rows;
    let entries = distinct + f64::from(u8::from(has_nulls));
    if optimistic_ratio(profile, sample, entries, best_ratio) <= best_ratio {
        return Admission::Reject;
    }

    if entries > MAX_DICTIONARY_ENTRIES as f64
        || distinct * profile.mean_value_bytes() > MAX_DICTIONARY_BYTES as f64
        || profile.value_bytes > MAX_DICTIONARY_INPUT_BYTES
        || profile.rows.saturating_mul(size_of::<u16>()) > MAX_DICTIONARY_CODE_BYTES
    {
        return Admission::TooLarge;
    }
    Admission::Trial
}

/// Estimates the compression ratio of a dictionary with `entries` values.
///
/// The model assumes dictionary values compress as well as the best estimate so far compresses
/// the whole input, and that codes compress to their bit width over the runs the probe saw.
fn optimistic_ratio(
    profile: &InputProfile,
    sample: &ProbeSample,
    entries: f64,
    values_ratio: f64,
) -> f64 {
    let values_bytes = entries * (profile.mean_value_bytes() + size_of::<BinaryView>() as f64);
    let code_bits = entries.max(2.0).log2().ceil();
    let code_bytes = profile.rows as f64 * sample.run_start_fraction() * code_bits / 8.0;
    profile.logical_nbytes as f64 / (values_bytes / values_ratio + code_bytes).max(1.0)
}

/// Builds a dictionary within the trial storage bounds, or returns `None` if it exceeds them.
fn build_candidate(
    array: &ArrayRef,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Option<DictArray>> {
    let constraints = DictConstraints {
        max_bytes: MAX_DICTIONARY_BYTES,
        max_len: MAX_DICTIONARY_ENTRIES,
    };
    let dictionary = dict_encode_with_constraints(array, &constraints, exec_ctx)?;
    Ok((dictionary.len() == array.len()).then_some(dictionary))
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
mod tests;
