// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BitPacking integer encoding.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::bitpack_compress::bit_width_histogram;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::bitpack_compress::find_best_bit_width;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::compress_patches;

/// BitPacking encoding for non-negative integers.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BitPackingScheme;

impl Scheme for BitPackingScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.bitpacking"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
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
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive_array = data.array_as_primitive();

        let histogram = bit_width_histogram(primitive_array, exec_ctx)?;
        let bw = find_best_bit_width(primitive_array.ptype(), &histogram)?;

        // If best bw is determined to be the current bit-width, return the original array.
        if bw as usize == primitive_array.ptype().bit_width() {
            return Ok(primitive_array.array().clone());
        }

        // Otherwise we can bitpack the array.
        let primitive_array = primitive_array.into_owned();
        let packed = bitpack_encode(&primitive_array, bw, Some(&histogram), exec_ctx)?;

        let packed_stats = packed.statistics().to_owned();
        let ptype = packed.dtype().as_ptype();
        let mut parts = BitPacked::into_parts(packed);

        let array = if use_experimental_patches() {
            let patches = parts.patches.take();
            // Transpose patches into G-ALP style PatchedArray, wrapping an inner BitPackedArray.
            let array = BitPacked::try_new(
                parts.packed,
                ptype,
                parts.validity,
                None,
                parts.bit_width,
                parts.len,
                parts.offset,
            )?
            .into_array();

            match patches {
                None => array,
                Some(p) => Patched::from_array_and_patches(array, &p, exec_ctx)?
                    .with_stats_set(packed_stats)
                    .into_array(),
            }
        } else {
            // Compress patches and place back into BitPackedArray.
            let patches = parts
                .patches
                .take()
                .map(|p| compress_patches(p, exec_ctx))
                .transpose()?;
            parts.patches = patches;
            BitPacked::try_new(
                parts.packed,
                ptype,
                parts.validity,
                parts.patches,
                parts.bit_width,
                parts.len,
                parts.offset,
            )?
            .with_stats_set(packed_stats)
            .into_array()
        };

        Ok(array)
    }
}
