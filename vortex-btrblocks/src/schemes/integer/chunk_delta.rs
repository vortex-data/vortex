// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lag-1 deltas restarting at every 1024-element chunk, less each chunk's minimum.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::ChunkDelta;
use vortex_fastlanes::ChunkDeltaArray;
use vortex_fastlanes::ChunkDeltaArrayExt;
use vortex_fastlanes::ChunkDeltaArraySlotsExt;
use vortex_fastlanes::FL_CHUNK_SIZE;

use super::DeltaScheme;
use super::RunEndScheme;
use super::ZigZagScheme;
use super::affine::sample_chunks;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::GenerateStatsOptions;
use crate::Scheme;
use crate::SchemeExt;

/// Encodes each value as its difference from the previous one less the chunk's smallest
/// difference, restarting at every chunk, as Parquet's `DELTA_BINARY_PACKED` does.
///
/// It keeps one base per chunk where [`DeltaScheme`](super::DeltaScheme) keeps one per SIMD lane,
/// and its residuals are never negative, so they cascade straight into bit-packing, ideally at
/// one width per chunk.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ChunkDeltaScheme;

impl Scheme for ChunkDeltaScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.delta.chunked"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![ChunkDelta.id()]
    }

    /// Children: deltas=0, bases=1, mins=2.
    fn num_children(&self) -> usize {
        3
    }

    /// The deltas are never negative, and differencing them again rarely pays.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        [ZigZagScheme.id(), self.id()]
            .into_iter()
            .map(|excluded| DescendantExclusion {
                excluded,
                children: ChildSelection::One(0),
            })
            .collect()
    }

    /// Dict codes have no order between neighbours.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        [
            IntDictScheme.id(),
            FloatDictScheme.id(),
            StringDictScheme.id(),
            BinaryDictScheme.id(),
        ]
        .into_iter()
        .map(|ancestor| AncestorExclusion {
            ancestor,
            children: ChildSelection::One(1),
        })
        .collect()
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // The deltas only shrink once a later layer packs them. Samples splice short runs from
        // across the array, so the deltas at their seams mean nothing.
        if compress_ctx.finished_cascading()
            || compress_ctx.is_sample()
            || data.array_len() < FL_CHUNK_SIZE
        {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        let scheme = *self;
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |compressor, data, _best_so_far, compress_ctx, exec_ctx| {
                let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
                // Whole chunks restart their deltas, so a sample of whole chunks is exact.
                let sample = sample_chunks(&primitive, exec_ctx)?;
                let encoded = ChunkDelta::encode(&sample, exec_ctx)?;
                let compressed =
                    compress_children(&scheme, &encoded, compressor, &compress_ctx, exec_ctx)?;
                let bytes = compressed.nbytes();
                // Delta and RunEnd estimate themselves from short runs or from packing alone, so
                // they look worse next to this full cascade than they are. Compress the same
                // sample with each and step aside if either does at least as well.
                let sample = sample.into_array();
                let rivals: [&dyn Scheme; 2] = [&DeltaScheme::default(), &RunEndScheme];
                for rival in rivals {
                    if !compressor.has_scheme(rival.id()) {
                        continue;
                    }
                    let data = ArrayAndStats::new(sample.clone(), GenerateStatsOptions::default());
                    if let Ok(theirs) = rival.compress(compressor, &data, compress_ctx.clone(), exec_ctx)
                        && theirs.nbytes() <= bytes
                    {
                        return Ok(EstimateVerdict::Skip);
                    }
                }
                let bits = bytes as f64 * 8.0 / sample.len() as f64;
                let full_width = primitive.ptype().bit_width() as f64;
                Ok(EstimateVerdict::Ratio(full_width / bits.max(1e-3)))
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
        let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
        let encoded = ChunkDelta::encode(&primitive, exec_ctx)?;
        compress_children(self, &encoded, compressor, &compress_ctx, exec_ctx)
    }
}

fn compress_children(
    scheme: &ChunkDeltaScheme,
    array: &ChunkDeltaArray,
    compressor: &CascadingCompressor,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let deltas =
        compressor.compress_child(array.deltas(), compress_ctx, scheme.id(), 0, exec_ctx)?;
    let bases =
        compressor.compress_child(array.bases(), compress_ctx, scheme.id(), 1, exec_ctx)?;
    let mins = compressor.compress_child(array.mins(), compress_ctx, scheme.id(), 2, exec_ctx)?;
    Ok(ChunkDelta::try_new(
        deltas,
        bases,
        mins,
        ChunkDeltaArrayExt::validity(array),
        array.len(),
        array.offset(),
    )?
    .into_array())
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_fastlanes::ChunkDelta;

    use super::ChunkDeltaScheme;
    use crate::BtrBlocksCompressorBuilder;
    use crate::SESSION;
    use crate::schemes::integer::VarBitPackingScheme;

    /// A random walk of small steps: the steps are tiny, but values far apart differ a lot.
    #[test]
    fn picks_lag_one_deltas_for_a_walk() -> VortexResult<()> {
        let mut v = 0i64;
        let mut state = 7u64;
        let values = PrimitiveArray::from_iter((0..64 * 1024).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v += (state >> 61) as i64 - 3;
            v
        }))
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let compressed = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .with_new_scheme(&ChunkDeltaScheme)
            .with_new_scheme(&VarBitPackingScheme)
            .build()
            .compress(&values, &mut ctx)?;
        assert!(compressed.is::<ChunkDelta>(), "{}", compressed.display_tree());
        assert_arrays_eq!(compressed, values, &mut ctx);
        // Steps of -3..=4 less the minimum fit three bits, plus the per-chunk overhead.
        assert!(compressed.nbytes() * 8 < 4 * values.len() as u64, "{}", compressed.nbytes());
        Ok(())
    }
}
