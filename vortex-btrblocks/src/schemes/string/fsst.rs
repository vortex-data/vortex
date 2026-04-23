// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST string compression scheme.
//!
//! FSST sampling is relatively expensive because it trains a symbol table and encodes sample
//! strings. The estimator in this module tries to avoid that work when simple string statistics
//! already make the decision clear. It first rejects shapes that another string scheme handles
//! better, such as all-null and null-dominated arrays. It then looks for the strongest cheap FSST
//! signal: many values sharing both a leading shape and a suffix-aware shape. Only that case uses
//! the structural size model to return an immediate ratio without requesting to compress a sample.
//!
//! Low prefix-cardinality columns are especially promising for FSST, but ambiguous with
//! dictionary-friendly repeated categories. String stats therefore track both coarse
//! `(length, first four bytes)` cardinality and suffix-aware `(length, first four bytes, last four
//! bytes)` cardinality. FSST uses the coarse signal to detect shared leading structure and the
//! suffix-aware signal to distinguish repeated full shapes from common-prefix/high-cardinality
//! values. Boundary cases still fall back to the normal sampling path.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::stats::GenerateStatsOptions;
use vortex_error::VortexResult;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;

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

    fn stats_options(&self) -> GenerateStatsOptions {
        GenerateStatsOptions {
            count_distinct_values: true,
        }
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
/// `StringStats::estimated_prefix_distinct_count` counts distinct `(length, first four bytes)` keys
/// over non-null values. A density at or below this value means many rows share a short leading
/// shape. That shape is promising for FSST, but also ambiguous with dictionary-friendly repeated
/// categories, so the estimator checks suffix-aware cardinality before applying its most aggressive
/// byte-code discounts.
const FSST_LOW_PREFIX_DENSITY: f64 = 0.125;

/// Suffix-aware density threshold that confirms repeated full string shapes.
///
/// `StringStats::estimated_distinct_count` counts distinct
/// `(length, first four bytes, last four bytes)` keys. Keeping this threshold equal to
/// [`FSST_LOW_PREFIX_DENSITY`] means a low-prefix-density candidate only receives the strongest
/// FSST byte-code discount when suffix-aware cardinality is just as low as prefix cardinality.
const FSST_LOW_SUFFIX_AWARE_DENSITY: f64 = 0.125;

/// Code-byte factor for candidates with repeated prefixes and suffix-aware shapes.
///
/// Very low `(length, prefix, suffix)` density usually means a generated or template-like string
/// shape. FSST tends to do well there because trained symbols cover the stable byte runs while only
/// the middle variation remains mostly literal.
const FSST_LOW_PREFIX_SUFFIX_CODE_BYTE_FACTOR: f64 = 0.12;

/// Estimate whether FSST is worth using without immediately compressing an FSST sample.
///
/// This estimator is intentionally not an exact prediction of the trained FSST output. Its job is
/// narrower: avoid the expensive sampling path when cheap statistics already give us a defensible
/// positive answer, and preserve sampling for inputs whose result depends on the actual trained
/// symbol table.
///
/// The algorithm has three stages:
///
/// 1. **Hard exclusions.** We skip all-null arrays defensively and skip arrays where more than
///    [`NULL_DOMINATED_SPARSE_MIN_RATIO`] of rows are null. The null-dominated sparse string scheme
///    has a stronger immediate signal for that shape and avoids training FSST over a tiny set of
///    non-null strings.
///
/// 2. **Repeated-shape gate.** String stats provide two approximate cardinality signals:
///    `(length, first four bytes)` prefix cardinality and
///    `(length, first four bytes, last four bytes)` suffix-aware cardinality. FSST only bypasses
///    sampling when both densities are very low. Low prefix density alone is ambiguous: it can
///    describe values with shared leading text and changing tails, but it can also describe
///    dictionary-friendly categories or values where the trained FSST table matters. Those
///    ambiguous shapes deliberately fall back to [`DeferredEstimate::Sample`].
///
/// 3. **Structural size model.** Once the repeated-shape gate passes, the estimator compares a
///    modeled FSST size to the current canonical `data.array().nbytes()` baseline. FSST replaces a
///    canonical `VarBinView` representation with compressed code bytes, a code-offset child, an
///    original-length child, and a bounded symbol table:
///
///    `after = total_value_bytes * code_byte_factor
///           + value_count * FSST_CHILD_METADATA_NBYTES_PER_VALUE
///           + FSST_MAX_SYMBOL_TABLE_NBYTES`
///
///    The only immediate positive path uses [`FSST_LOW_PREFIX_SUFFIX_CODE_BYTE_FACTOR`], because
///    both cheap cardinality signals agree that the column has repeated whole-string shape. The
///    model no longer chooses FSST from row-view savings alone; that was too eager for file-size
///    sensitive selection because it skipped the normal sampled comparison against dictionary and
///    other string schemes.
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
        .estimated_prefix_distinct_count()
        .map(|count| f64::from(count) / f64::from(stats.value_count()));

    if matches!(prefix_density, Some(density) if density <= FSST_LOW_PREFIX_DENSITY) {
        let value_count = stats.value_count();
        let suffix_aware_density = stats
            .estimated_distinct_count()
            .map(|count| f64::from(count) / f64::from(value_count));

        if suffix_aware_density.is_none_or(|density| density > FSST_LOW_SUFFIX_AWARE_DENSITY) {
            return CompressionEstimate::Deferred(DeferredEstimate::Sample);
        }

        let estimated_ratio = fsst_estimated_ratio(
            data.array().nbytes(),
            stats.total_value_bytes(),
            value_count,
            FSST_LOW_PREFIX_SUFFIX_CODE_BYTE_FACTOR,
        );

        if estimated_ratio.is_finite() && estimated_ratio >= FSST_MIN_FAST_PATH_RATIO {
            return CompressionEstimate::Verdict(EstimateVerdict::Ratio(estimated_ratio));
        }

        return CompressionEstimate::Deferred(DeferredEstimate::Sample);
    }

    CompressionEstimate::Deferred(DeferredEstimate::Sample)
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

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_compressor::stats::GenerateStatsOptions;

    use super::*;

    fn estimate_for(values: Vec<Option<String>>) -> CompressionEstimate {
        let array = VarBinViewArray::from_iter(values, DType::Utf8(Nullability::NonNullable));
        let data = ArrayAndStats::new(
            array.into_array(),
            GenerateStatsOptions {
                count_distinct_values: true,
            },
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        estimate_fsst_compression_ratio(&data, &mut ctx)
    }

    #[test]
    fn fsst_fast_path_requires_low_suffix_aware_density() {
        let values = (0..4096)
            .map(|idx| Some(format!("acct-{idx:08}")))
            .collect();

        assert!(matches!(
            estimate_for(values),
            CompressionEstimate::Deferred(DeferredEstimate::Sample)
        ));
    }

    #[test]
    fn fsst_fast_path_allows_low_prefix_and_suffix_aware_density() {
        let values = (0..4096)
            .map(|idx| Some(format!("acct-{idx:08}-tail")))
            .collect();

        assert!(matches!(
            estimate_for(values),
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(_))
        ));
    }
}
