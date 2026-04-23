// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST string compression scheme.
//!
//! FSST sampling is relatively expensive because it trains a symbol table and encodes sample
//! strings. The estimator in this module tries to avoid that work when simple string statistics
//! already make the decision clear. It first rejects shapes that another string scheme handles
//! better, such as all-null and null-dominated arrays. It then builds a conservative structural
//! model of an FSST array: compressed code bytes, compressed length/offset children, and bounded
//! symbol-table overhead. If that model predicts a clear win, the scheme returns an immediate
//! ratio without requesting to compress a sample.
//!
//! The only delayed path is for low prefix-cardinality columns. That signal is promising for FSST,
//! but ambiguous with dictionary-friendly repeated categories, so the estimator schedules a cheap
//! callback that scans full strings once and counts `(length, first four bytes, last four bytes)`
//! keys. This suffix-aware repetition probe is still much cheaper than FSST sampling, and it either
//! returns a stronger ratio for clearly repeated formats or skips FSST if it cannot beat the best
//! estimate so far. Boundary cases still fall back to the normal sampling path.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_utils::aliases::hash_set::HashSet;

use super::NULL_DOMINATED_SPARSE_MIN_RATIO;
use super::is_utf8_string;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// FSST (Fast Static Symbol Table) compression.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FSSTScheme;

impl Scheme for FSSTScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.fsst"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        is_utf8_string(canonical)
    }

    /// Children: lengths=0, code_offsets=1.
    fn num_children(&self) -> usize {
        2
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        estimate_fsst_compression_ratio(data, exec_ctx)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let utf8 = data.array_as_utf8().into_owned();
        let compressor_fsst = fsst_train_compressor(&utf8);
        let fsst = fsst_compress(&utf8, utf8.len(), utf8.dtype(), &compressor_fsst, exec_ctx);

        let uncompressed_lengths_primitive = fsst
            .uncompressed_lengths()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow()?;
        let compressed_original_lengths = compressor.compress_child(
            &uncompressed_lengths_primitive.into_array(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;

        let codes_offsets_primitive = fsst
            .codes()
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow()?;
        let compressed_codes_offsets = compressor.compress_child(
            &codes_offsets_primitive.into_array(),
            &compress_ctx,
            self.id(),
            1,
            exec_ctx,
        )?;
        let compressed_codes = VarBinArray::try_new(
            compressed_codes_offsets,
            fsst.codes().bytes().clone(),
            fsst.codes().dtype().clone(),
            fsst.codes().validity()?,
        )?;

        let fsst = FSST::try_new(
            fsst.dtype().clone(),
            fsst.symbols().clone(),
            fsst.symbol_lengths().clone(),
            compressed_codes,
            compressed_original_lengths,
            exec_ctx,
        )?;

        Ok(fsst.into_array())
    }
}

/// Conservative upper bound for the FSST symbol table metadata.
///
/// FSST can emit at most 255 symbols. Vortex stores each symbol as the FSST `Symbol` payload
/// (8 bytes) plus a one-byte symbol length, so the table is bounded by `255 * 9` bytes. The
/// estimator treats this as fully uncompressed overhead even though the symbol table is small and
/// fixed-size. That makes the fast path less likely to choose FSST for tiny arrays where the
/// symbol table dominates the result.
const FSST_MAX_SYMBOL_TABLE_NBYTES: f64 = 255.0 * 9.0;

/// Estimated post-cascade metadata cost per non-null string value.
///
/// FSST replaces VarBinView's 16-byte per-row views with two integer children: original string
/// lengths and compressed-code offsets. Those children are monotonic/small integer arrays that are
/// recursively compressed by btrblocks. This constant is an empirical budget for their combined
/// post-cascade footprint, not their raw in-memory width. We intentionally keep it separate from
/// the byte-code estimate so the model can distinguish structural metadata savings from actual
/// FSST byte-code compression.
const FSST_CHILD_METADATA_NBYTES_PER_VALUE: f64 = 2.0;

/// Minimum estimated ratio required before FSST bypasses sampling.
///
/// The estimator uses cheap byte-count and prefix/suffix signals, so it keeps a small margin above
/// `1.0` before returning an immediate ratio. Boundary cases continue to use the existing sampling
/// path, which is slower but safer when the structural model predicts only a marginal win.
const FSST_MIN_FAST_PATH_RATIO: f64 = 1.02;

/// Prefix-density threshold that marks a string column as structurally repetitive.
///
/// `StringStats::estimated_distinct_count` counts distinct `(length, first four bytes)` keys over
/// non-null values. A density at or below this value means many rows share a short leading shape.
/// That shape is promising for FSST, but also ambiguous with dictionary-friendly repeated
/// categories, so the estimator uses this threshold to trigger the delayed suffix probe instead of
/// immediately accepting or rejecting FSST.
const FSST_LOW_PREFIX_DENSITY: f64 = 0.125;

/// Prefix-density threshold for a cheap immediate FSST byte-code discount.
///
/// Densities above [`FSST_LOW_PREFIX_DENSITY`] but at or below this value still indicate useful
/// repetition in leading bytes. They are less ambiguous than the very-low-density case, so the
/// estimator can apply a conservative code-byte factor directly without scheduling the delayed
/// suffix probe.
const FSST_MODERATE_PREFIX_DENSITY: f64 = 0.5;

/// Suffix-density threshold that confirms repeated full string shapes in the delayed probe.
///
/// The delayed callback counts distinct `(length, first four bytes, last four bytes)` keys. Keeping
/// this threshold equal to [`FSST_LOW_PREFIX_DENSITY`] means a low-prefix-density candidate only
/// receives the strongest FSST byte-code discount when suffixes are just as repetitive as prefixes.
const FSST_LOW_SUFFIX_DENSITY: f64 = 0.125;

/// Average value length cutoff for short, repeated string formats.
///
/// When prefix density is low but suffix density is not low, the column may contain generated
/// identifiers, path-like strings, URLs, or category values with a shared prefix and changing tail.
/// Shorter values leave less room for an FSST-trained symbol table to amortize overhead, so they
/// get a more aggressive byte-code discount than long values only after the delayed probe has
/// proved the column is repetitive enough to be worth considering.
const FSST_SHORT_AVERAGE_VALUE_NBYTES: f64 = 96.0;

/// Code-byte factor used when no cheap repetition signal is available.
///
/// A factor of `1.0` means the structural model assumes FSST emits as many code bytes as the input
/// has value bytes. This still lets FSST win on large arrays where removing VarBinView row views and
/// recursively compressing length/offset children outweighs symbol-table overhead.
const FSST_UNCOMPRESSED_CODE_BYTE_FACTOR: f64 = 1.0;

/// Code-byte factor for moderate prefix repetition.
///
/// This is intentionally close to `1.0`: it represents a conservative discount for columns whose
/// leading bytes repeat, while still leaving borderline cases to sampling if the structural savings
/// are not enough to clear [`FSST_MIN_FAST_PATH_RATIO`].
const FSST_MODERATE_PREFIX_CODE_BYTE_FACTOR: f64 = 0.8;

/// Code-byte factor for delayed-probe candidates with repeated prefixes and suffixes.
///
/// Very low `(length, prefix, suffix)` density usually means a generated or template-like string
/// shape. FSST tends to do well there because trained symbols cover the stable byte runs while only
/// the middle variation remains mostly literal.
const FSST_LOW_PREFIX_SUFFIX_CODE_BYTE_FACTOR: f64 = 0.12;

/// Code-byte factor for short repeated formats that only pass the prefix-density probe.
///
/// This applies when the prefix is highly repetitive but the suffix has enough variation that we do
/// not want the strongest discount. It is still more optimistic than the moderate-prefix immediate
/// factor because the delayed callback has scanned the whole column and ruled out purely tiny-sample
/// artifacts.
const FSST_SHORT_REPEATED_CODE_BYTE_FACTOR: f64 = 0.25;

/// Code-byte factor for long repeated formats that only pass the prefix-density probe.
///
/// Long values tend to have more unique payload bytes after the shared prefix/suffix structure, so
/// the delayed estimator uses a weaker discount for them and lets sampling handle borderline cases.
const FSST_LONG_REPEATED_CODE_BYTE_FACTOR: f64 = 0.5;

/// Extra repetition information used by the delayed low-prefix-cardinality FSST estimator.
#[derive(Debug)]
struct FSSTRepetitionStats {
    /// Number of distinct `(length, first four bytes, last four bytes)` keys among non-null values.
    len_prefix_suffix_distinct_count: usize,
}

/// Estimate whether FSST is worth using without immediately compressing an FSST sample.
///
/// This estimator is intentionally not an exact prediction of the trained FSST output. Its job is
/// narrower: avoid the expensive sampling path when cheap statistics already give us a defensible
/// answer, and preserve sampling for inputs whose result depends on the actual trained symbol
/// table.
///
/// The algorithm has three stages:
///
/// 1. **Hard exclusions.** We skip all-null arrays defensively and skip arrays where more than
///    [`NULL_DOMINATED_SPARSE_MIN_RATIO`] of rows are null. The null-dominated sparse string scheme
///    has a stronger immediate signal for that shape and avoids training FSST over a tiny set of
///    non-null strings.
///
/// 2. **Structural size model.** FSST replaces a canonical `VarBinView` representation with:
///    compressed code bytes, a code-offset child, an original-length child, and a bounded symbol
///    table. The estimator reads three cheap values from string stats: total value bytes,
///    non-null value count, and approximate distinct count. It compares the modeled FSST size to
///    the current canonical `data.array().nbytes()` baseline using:
///
///    `after = total_value_bytes * code_byte_factor
///           + value_count * FSST_CHILD_METADATA_NBYTES_PER_VALUE
///           + FSST_MAX_SYMBOL_TABLE_NBYTES`
///
///    The default [`FSST_UNCOMPRESSED_CODE_BYTE_FACTOR`] means "assume FSST does not compress the
///    bytes at all." If that still beats the canonical size, FSST is a good structural win because
///    it removes expensive VarBinView row views and lets the length/offset children cascade through
///    integer compression. A moderate prefix-density signal lowers the factor to
///    [`FSST_MODERATE_PREFIX_CODE_BYTE_FACTOR`], reflecting likely symbol reuse while still staying
///    conservative.
///
/// 3. **Delayed repetition probe for low prefix density.** The approximate distinct count is based
///    on `(length, first four bytes)`. That is cheap, but it intentionally under-counts
///    common-prefix data. Low prefix density can mean excellent FSST input, such as generated URLs,
///    UUID-like fixed formats, or common prefix/suffix strings, but it can also mean
///    dictionary-friendly repeated categories. For this ambiguous but promising case, we return a
///    deferred callback that scans the full strings once and counts distinct
///    `(length, first four bytes, last four bytes)`. This is still much cheaper than sample
///    compression because it does not train an FSST table or encode any bytes. The callback uses
///    suffix density and average value length to choose one of the delayed-probe code-byte factors,
///    and it skips FSST when the modeled ratio cannot beat the current `best_so_far` estimate.
///
/// Any case whose modeled ratio is below [`FSST_MIN_FAST_PATH_RATIO`] falls back to
/// [`DeferredEstimate::Sample`]. Those are the boundary cases where the actual FSST symbol table
/// and byte distribution matter enough that sampling is still the safest choice.
fn estimate_fsst_compression_ratio(
    data: &ArrayAndStats,
    exec_ctx: &mut ExecutionCtx,
) -> CompressionEstimate {
    let stats = data.string_stats(exec_ctx);

    if stats.value_count() == 0 {
        return CompressionEstimate::Verdict(EstimateVerdict::Skip);
    }

    // Sparse handles this case with a stronger immediate estimate and without training FSST.
    let len = data.array_len() as f64;
    if f64::from(stats.null_count()) / len > NULL_DOMINATED_SPARSE_MIN_RATIO {
        return CompressionEstimate::Verdict(EstimateVerdict::Skip);
    }

    let prefix_density = stats
        .estimated_distinct_count()
        .map(|count| f64::from(count) / f64::from(stats.value_count()));

    if matches!(prefix_density, Some(density) if density <= FSST_LOW_PREFIX_DENSITY) {
        let before_nbytes = data.array().nbytes();
        let total_value_bytes = stats.total_value_bytes();
        let value_count = stats.value_count();

        return CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |_compressor, data, best_so_far, _ctx, _exec_ctx| {
                let repetition_stats = data.get_or_insert_with(|| fsst_repetition_stats(data));
                let suffix_density = repetition_stats.len_prefix_suffix_distinct_count as f64
                    / f64::from(value_count);
                let average_value_bytes = total_value_bytes as f64 / f64::from(value_count);
                let code_byte_factor = if suffix_density <= FSST_LOW_SUFFIX_DENSITY {
                    FSST_LOW_PREFIX_SUFFIX_CODE_BYTE_FACTOR
                } else if average_value_bytes <= FSST_SHORT_AVERAGE_VALUE_NBYTES {
                    FSST_SHORT_REPEATED_CODE_BYTE_FACTOR
                } else {
                    FSST_LONG_REPEATED_CODE_BYTE_FACTOR
                };
                let estimated_ratio = fsst_estimated_ratio(
                    before_nbytes,
                    total_value_bytes,
                    value_count,
                    code_byte_factor,
                );
                let does_not_beat_best = best_so_far
                    .and_then(|score| score.finite_ratio())
                    .is_some_and(|best_ratio| estimated_ratio <= best_ratio);

                if estimated_ratio.is_finite()
                    && estimated_ratio >= FSST_MIN_FAST_PATH_RATIO
                    && !does_not_beat_best
                {
                    Ok(EstimateVerdict::Ratio(estimated_ratio))
                } else {
                    Ok(EstimateVerdict::Skip)
                }
            },
        )));
    }

    let code_byte_factor = match prefix_density {
        Some(density) if density <= FSST_MODERATE_PREFIX_DENSITY => {
            FSST_MODERATE_PREFIX_CODE_BYTE_FACTOR
        }
        _ => FSST_UNCOMPRESSED_CODE_BYTE_FACTOR,
    };

    let estimated_ratio = fsst_estimated_ratio(
        data.array().nbytes(),
        stats.total_value_bytes(),
        stats.value_count(),
        code_byte_factor,
    );

    if estimated_ratio.is_finite() && estimated_ratio >= FSST_MIN_FAST_PATH_RATIO {
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(estimated_ratio))
    } else {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }
}

/// Compute the delayed repetition signal used for low-prefix-cardinality FSST candidates.
fn fsst_repetition_stats(data: &ArrayAndStats) -> FSSTRepetitionStats {
    let strings = data.array_as_utf8().into_owned();
    let mut distinct = HashSet::with_capacity(strings.len() / 2);

    strings.with_iterator(|iter| {
        iter.flatten().for_each(|value| {
            distinct.insert((value.len(), first_four(value), last_four(value)));
        });
    });

    FSSTRepetitionStats {
        len_prefix_suffix_distinct_count: distinct.len(),
    }
}

/// Return the first four bytes of a string, zero-padding shorter strings.
fn first_four(value: &[u8]) -> [u8; 4] {
    let mut prefix = [0u8; 4];
    let prefix_len = value.len().min(prefix.len());
    prefix[..prefix_len].copy_from_slice(&value[..prefix_len]);
    prefix
}

/// Return the last four bytes of a string, zero-padding shorter strings.
fn last_four(value: &[u8]) -> [u8; 4] {
    let mut suffix = [0u8; 4];
    let suffix_len = value.len().min(suffix.len());
    suffix[..suffix_len].copy_from_slice(&value[value.len() - suffix_len..]);
    suffix
}

/// Estimate the compression ratio from the FSST structural size model.
fn fsst_estimated_ratio(
    before_nbytes: u64,
    total_value_bytes: u64,
    value_count: u32,
    code_byte_factor: f64,
) -> f64 {
    let estimated_after_nbytes = total_value_bytes as f64 * code_byte_factor
        + f64::from(value_count) * FSST_CHILD_METADATA_NBYTES_PER_VALUE
        + FSST_MAX_SYMBOL_TABLE_NBYTES;

    before_nbytes as f64 / estimated_after_nbytes
}
