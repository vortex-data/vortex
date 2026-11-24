// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! BitPacking integer encoding.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateVerdict;
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
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct BitPackingScheme;

impl Scheme for BitPackingScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.bitpacking"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        // `compress_patches` stores constant patch values as a Constant array.
        let mut encodings = vec![BitPacked.id(), Constant.id()];
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
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive_array = compressor
            .with_compressed_validity(data.array(), exec_ctx)?
            .execute::<PrimitiveArray>(exec_ctx)?;

        let histogram = bit_width_histogram(primitive_array.as_view(), exec_ctx)?;
        let bw = find_best_bit_width(primitive_array.ptype(), &histogram)?;
        if usize::from(bw) == primitive_array.ptype().bit_width() {
            return Ok(primitive_array.into_array());
        }

        let packed = bitpack_encode(&primitive_array, bw, Some(&histogram), exec_ctx)?;
        let packed_stats = packed.statistics().to_owned();
        let ptype = packed.dtype().as_ptype();
        let mut parts = BitPacked::into_parts(packed);

        let patches = if use_experimental_patches() {
            parts.patches.take()
        } else {
            parts.patches = parts
                .patches
                .take()
                .map(|p| compress_patches(p, exec_ctx))
                .transpose()?;
            None
        };

        let array = BitPacked::try_new(
            parts.packed,
            ptype,
            parts.validity,
            parts.patches,
            parts.bit_width,
            parts.len,
            parts.offset,
        )?;

        match patches {
            None => Ok(array.with_stats_set(packed_stats).into_array()),
            Some(p) => Ok(
                Patched::from_array_and_patches(array.into_array(), &p, exec_ctx)?
                    .with_stats_set(packed_stats)
                    .into_array(),
            ),
        }
    }
}
