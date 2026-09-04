// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BitPacking integer encoding.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Patched;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::BitPackedArraySlotsExt;
use vortex_fastlanes::BitPackedSlots;
use vortex_fastlanes::bitpack_compress::bit_width_histogram;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_chunk_widths;
use vortex_fastlanes::bitpack_compress::find_best_bit_width;
use vortex_fastlanes::bitpacked_v2_id;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::compress_patches;

/// BitPacking encoding for non-negative integers.
///
/// Every 1024-element chunk gets its own bit width when the compressor may use the
/// `fastlanes.bitpacked_v2` format. Otherwise every chunk shares one width, which is the original
/// `fastlanes.bitpacked` format.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BitPackingScheme;

impl Scheme for BitPackingScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.bitpacking"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        let mut encodings = vec![BitPacked.id()];
        if use_experimental_patches() {
            encodings.push(Patched.id());
        }
        encodings
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let stats = data.integer_stats(exec_ctx);

        // BitPacking only works for non-negative values.
        if stats.erased().min_is_negative() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive_array = data.array_as_primitive();
        let full_width = primitive_array.ptype().bit_width();

        // Per-chunk widths are the newer format of BitPacked. Produce them whenever the writer
        // may serialize them; otherwise every chunk shares one width, the original format.
        let packed = if compressor.allows_serialized_id(bitpacked_v2_id()) {
            bitpack_to_best_chunk_widths(&data.array_as_primitive().into_owned(), exec_ctx)?
        } else {
            let histogram = bit_width_histogram(primitive_array, exec_ctx)?;
            let bw = find_best_bit_width(primitive_array.ptype(), &histogram)?;
            if bw as usize == full_width {
                return Ok(primitive_array.array().clone());
            }
            bitpack_encode(
                &data.array_as_primitive().into_owned(),
                bw,
                Some(&histogram),
                exec_ctx,
            )?
        };

        // If every chunk needs the full bit-width, return the original array.
        if packed.chunk_widths().uniform_width().map(usize::from) == Some(full_width) {
            return Ok(primitive_array.array().clone());
        }
        // Mostly patches means the values were never really packed: the array is sparse in all
        // but name, and beats raw storage only by the bytes of the few values that did fit.
        if packed
            .patches()
            .is_some_and(|p| p.num_patches() * 2 >= packed.len())
        {
            return Ok(primitive_array.array().clone());
        }

        let packed_stats = packed.statistics().to_owned();
        let ptype = packed.dtype().as_ptype();
        let mut parts = BitPacked::into_parts(packed);
        let patches = parts.patches.take();

        if use_experimental_patches() {
            // Transpose patches into G-ALP style PatchedArray, wrapping an inner BitPackedArray.
            let array = BitPacked::try_new(
                parts.packed,
                ptype,
                parts.validity,
                None,
                parts.widths,
                parts.len,
                parts.offset,
            )?;
            let array =
                compress_width_table(compressor, array, &compress_ctx, exec_ctx)?.into_array();
            return Ok(match patches {
                None => array,
                Some(p) => Patched::from_array_and_patches(array, &p, exec_ctx)?
                    .with_stats_set(packed_stats)
                    .into_array(),
            });
        }

        // Compress patches and place back into BitPackedArray.
        let patches = patches.map(|p| compress_patches(p, exec_ctx)).transpose()?;
        let array = BitPacked::try_new(
            parts.packed,
            ptype,
            parts.validity,
            patches,
            parts.widths,
            parts.len,
            parts.offset,
        )?;
        Ok(
            compress_width_table(compressor, array, &compress_ctx, exec_ctx)?
                .with_stats_set(packed_stats)
                .into_array(),
        )
    }
}

/// Re-encode the width table child through the compressor. Arrays whose chunks share one width
/// have no table.
fn compress_width_table(
    compressor: &CascadingCompressor,
    packed: BitPackedArray,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<BitPackedArray> {
    let Some(table) = packed.width_table().cloned() else {
        return Ok(packed);
    };
    let compressed = compressor.compress_child(
        &table,
        compress_ctx,
        BitPackingScheme.id(),
        BitPackedSlots::WIDTH_TABLE,
        exec_ctx,
    )?;
    BitPacked::with_width_table(packed, compressed)
}
