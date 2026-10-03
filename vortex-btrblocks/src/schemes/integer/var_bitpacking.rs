// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bit-packing with one bit width per 1024-element chunk.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::VarBitPacked;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

/// Per-chunk widths must save at least this many bits per value over one width for the whole
/// array, or plain [`BitPackingScheme`](super::BitPackingScheme), which has more compute kernels,
/// is preferred.
const MIN_BITS_SAVED_PER_VALUE: f64 = 0.5;

/// Bit-packs each 1024-element chunk at the width of its own largest value.
///
/// This suits residuals whose range drifts across an array, such as those of [`Affine`]'s
/// per-chunk models, where one wide chunk would otherwise set the width for every chunk.
///
/// [`Affine`]: vortex_fastlanes::Affine
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct VarBitPackingScheme;

impl Scheme for VarBitPackingScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.bitpacking.chunked"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![VarBitPacked.id()]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // Samples splice short runs from across the array, so they say nothing about the
        // widths of whole chunks.
        if compress_ctx.is_sample() || data.array_len() < FL_CHUNK_SIZE {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        if data.integer_stats(exec_ctx).erased().min_is_negative() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            |_compressor, data, _best_so_far, _compress_ctx, _exec_ctx| {
                let primitive = data.array_as_primitive();
                let ptype = primitive.ptype().to_unsigned();
                let widths = match_each_unsigned_integer_ptype!(ptype, |U| {
                    let values = primitive.reinterpret_cast(ptype);
                    chunk_widths(values.as_slice::<U>())
                });
                let len = primitive.len() as f64;
                let max = f64::from(widths.iter().copied().max().unwrap_or(0));
                let packed = widths.iter().map(|&w| f64::from(w)).sum::<f64>()
                    * FL_CHUNK_SIZE as f64
                    + 8.0 * widths.len() as f64;
                let bits = packed / len;
                if max - bits < MIN_BITS_SAVED_PER_VALUE {
                    return Ok(EstimateVerdict::Skip);
                }
                Ok(EstimateVerdict::Ratio(
                    ptype.bit_width() as f64 / bits.max(1e-3),
                ))
            },
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = data.array_as_primitive().into_owned();
        Ok(VarBitPacked::encode(&primitive, exec_ctx)?.into_array())
    }
}

/// The bit width of each chunk's largest value. Null positions count with whatever they hold, so
/// this can only overestimate.
fn chunk_widths<U: Copy + Into<u64>>(values: &[U]) -> Vec<u8> {
    values
        .chunks(FL_CHUNK_SIZE)
        .map(|chunk| {
            let max = chunk.iter().fold(0u64, |m, &v| m | v.into());
            (u64::BITS - max.leading_zeros()) as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_fastlanes::VarBitPacked;

    use super::VarBitPackingScheme;
    use crate::BtrBlocksCompressorBuilder;
    use crate::SESSION;

    /// Fifteen chunks of two-bit values and one chunk of 30-bit values: one width for the whole
    /// array either wastes 28 bits on most values or patches a whole chunk.
    #[test]
    fn packs_each_chunk_at_its_own_width() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(3);
        let values = PrimitiveArray::from_iter((0..16 * 1024).map(|i| {
            if i < 15 * 1024 {
                rng.random_range(0..4u32)
            } else {
                rng.random_range(0..1u32 << 30)
            }
        }))
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let compressed = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .with_new_scheme(&VarBitPackingScheme)
            .build()
            .compress(&values, &mut ctx)?;
        assert!(compressed.is::<VarBitPacked>(), "{}", compressed.display_tree());
        assert_arrays_eq!(compressed, values, &mut ctx);
        Ok(())
    }
}
