// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse encoding for null-dominated string arrays.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::PType;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt as _;

use super::FSSTScheme;
use super::OnPairScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::Child;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::integer::IntDictScheme;
use crate::schemes::integer::SparseScheme as IntSparseScheme;

/// The number of consecutive values in each sampled run of values.
const SAMPLE_RUN_LEN: usize = 64;

/// The number of runs of values sampled to estimate how well the values compress.
const SAMPLE_RUNS: usize = 16;

/// Sparse encoding for null-dominated arrays.
///
/// This is the same as the integer `SparseScheme`, but we only use this for null-dominated arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NullDominatedSparseScheme;

impl Scheme for NullDominatedSparseScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.sparse"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Sparse.id()]
    }

    /// Children: values=0, indices=1.
    fn num_children(&self) -> usize {
        2
    }

    /// Sparse indices (child 1) are monotonically increasing positions with all unique values, so
    /// Sparse and Dict are pointless on them.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntSparseScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let len = data.array_len();
        let stats = data.varbinview_stats(exec_ctx);
        let value_count = stats.value_count() as usize;

        // All-null arrays should be compressed as constant instead anyways.
        if value_count == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Only null-dominated arrays (90% null) go this route.
        if stats.null_count() as f64 / len as f64 <= 0.9 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Sparse drops the views of null rows and adds an index per value.
        let Ok(before) = uncompressed_size_in_bytes(data.array(), exec_ctx) else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        let null_views = (len - value_count) * size_of::<BinaryView>();
        let values = before.saturating_sub(null_views);
        let indices = value_count * sparse_index_width(len);

        // Without a cascade the values keep all their bytes, so long values bound the ratio well
        // below the fraction of rows that are null.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Ratio(
                before as f64 / (values + indices) as f64,
            ));
        }

        // Otherwise the values shrink by whatever the other string schemes achieve on them, which
        // only compressing some of them can tell.
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |compressor, data, best_so_far, compress_ctx, exec_ctx| {
                // The indices alone bound the ratio, however well the values compress.
                let max_ratio = before as f64 / indices as f64;
                let threshold = best_so_far.and_then(EstimateScore::finite_ratio);
                if threshold.is_some_and(|threshold| max_ratio <= threshold) {
                    return Ok(EstimateVerdict::Skip);
                }

                let sample = sample_values(data.array(), exec_ctx)?;
                let sample_before = uncompressed_size_in_bytes(&sample, exec_ctx)?;
                // Mark subsampled values as a sample so constant detection cannot falsely collapse
                // a uniform subsample and sample-aware string trainers use their sample settings.
                let child_ctx = if sample.len() < value_count {
                    compress_ctx.with_sampling()
                } else {
                    compress_ctx
                };
                let compressed_sample = compressor.compress_child(
                    &sample,
                    &child_ctx,
                    NullDominatedSparseScheme.id(),
                    0,
                    exec_ctx,
                )?;
                let sample_after = compressed_sample.nbytes();
                if sample_before == 0 || sample_after == 0 {
                    return Ok(EstimateVerdict::Skip);
                }

                // Fixed tables (such as an FSST symbol table or OnPair dictionary) are paid once
                // across all valid values rather than scaling with the subsample ratio.
                let fixed = FSSTScheme
                    .fixed_nbytes(&compressed_sample)
                    .max(OnPairScheme::default().fixed_nbytes(&compressed_sample))
                    .min(sample_after);
                let variable = sample_after - fixed;
                let compressed_values =
                    fixed as f64 + values as f64 * variable as f64 / sample_before as f64;
                Ok(EstimateVerdict::Ratio(
                    before as f64 / (compressed_values + indices as f64),
                ))
            },
        )))
    }

    /// Children: values=0, indices=1. The values are the valid strings alone, so the other string
    /// schemes apply to them.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        // We pass None as we only run this pathway for NULL-dominated string arrays.
        let sparse_encoded = Sparse::encode(data.array(), None, exec_ctx)?;
        let Some(sparse) = sparse_encoded.as_opt::<Sparse>() else {
            return Ok(Encoded::leaf(sparse_encoded));
        };

        let indices = sparse
            .patches()
            .indices()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        let len = sparse.len();
        let fill = sparse.fill_scalar().clone();
        Ok(Encoded::new(
            vec![
                Child::new(self, 0, sparse.patches().values().clone()),
                Child::new(self, 1, indices.into_array()),
            ],
            Vec::new(),
            move |children, exec_ctx| {
                let [values, indices] = children.compressed()?;
                Sparse::try_new(indices, values, len, fill, exec_ctx).map(|a| a.into_array())
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

/// Byte width of narrowed 0-based patch indices (`0..len`).
fn sparse_index_width(len: usize) -> usize {
    PType::min_unsigned_ptype_for_value(len.saturating_sub(1) as u64).byte_width()
}

/// Returns evenly spaced runs of the valid values of `array`, or all of them if they are few.
fn sample_values(array: &ArrayRef, exec_ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let mask = array.validity()?.execute_mask(array.len(), exec_ctx)?;
    let values = array.filter(mask)?;
    let count = values.len();
    if count <= SAMPLE_RUNS * SAMPLE_RUN_LEN {
        return Ok(values);
    }

    let runs = (0..SAMPLE_RUNS)
        .map(|run| {
            let start = run * count / SAMPLE_RUNS;
            values.slice(start..start + SAMPLE_RUN_LEN)
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(ChunkedArray::try_new(runs, values.dtype().clone())?.into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    use rstest::rstest;
    use vortex_array::ArrayId;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_compressor::scheme::CompressionEstimate;
    use vortex_compressor::scheme::EstimateVerdict;
    use vortex_error::VortexResult;
    use vortex_sparse::Sparse;

    use super::NullDominatedSparseScheme;
    use super::sparse_index_width;
    use crate::ArrayAndStats;
    use crate::CascadingCompressor;
    use crate::CompressorContext;
    use crate::SESSION;
    use crate::Scheme;

    static SAW_SAMPLE_CTX: AtomicBool = AtomicBool::new(false);

    /// Probe string scheme that records whether it was evaluated under a sample context.
    #[derive(Debug)]
    struct SampleProbeScheme;

    impl Scheme for SampleProbeScheme {
        fn scheme_name(&self) -> &'static str {
            "vortex.test.string.sample_probe"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            canonical.dtype().is_utf8()
        }

        fn produced_encodings(&self) -> Vec<ArrayId> {
            Vec::new()
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            if compress_ctx.is_sample() {
                SAW_SAMPLE_CTX.store(true, Ordering::Relaxed);
            }
            CompressionEstimate::Verdict(EstimateVerdict::Skip)
        }

        #[expect(clippy::unimplemented, reason = "never selected")]
        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            unimplemented!()
        }
    }

    #[rstest]
    #[case::u8_max(256, 1)]
    #[case::u16_min(257, 2)]
    #[case::u16_max(65_536, 2)]
    #[case::u32_min(65_537, 4)]
    fn test_string_sparse_index_width_boundaries(
        #[case] len: usize,
        #[case] expected_width: usize,
    ) {
        assert_eq!(sparse_index_width(len), expected_width);
    }

    #[test]
    fn test_string_sparse_callback_marks_subsampled_values_as_sample() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        SAW_SAMPLE_CTX.store(false, Ordering::Relaxed);

        // 40,000 rows, 95% null -> 2,000 valid strings (> 1,024 sample size), where every
        // sampled run holds the exact same string while unsampled positions differ. Without
        // `with_sampling()`, constant detection collapses the sample before `SampleProbeScheme`
        // ever runs and inflates the estimated ratio.
        let valid_count = 2_000usize;
        let mut valid_seen = 0usize;
        let input = VarBinViewArray::from_iter(
            (0..40_000).map(|i| {
                (i % 20 == 0).then(|| {
                    let v = valid_seen;
                    valid_seen += 1;
                    // `sample_values` samples 16 runs of 64 starting at `run * 2000 / 16 = run * 125`.
                    let in_sampled_run = (v % 125) < 64;
                    if in_sampled_run {
                        "identical-sampled-string".to_owned()
                    } else {
                        format!("unique-unsampled-string-{v:06}")
                    }
                })
            }),
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        assert_eq!(valid_seen, valid_count);

        let compressor =
            CascadingCompressor::new(vec![&NullDominatedSparseScheme, &SampleProbeScheme]);
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert!(
            SAW_SAMPLE_CTX.load(Ordering::Relaxed),
            "expected string sparse callback to pass a sampling CompressorContext"
        );
        assert!(compressed.is::<Sparse>());
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }
}
