// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run-length integer encoding and shared RLE compression helpers.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::DescendantExclusion;
#[cfg(feature = "unstable_encodings")]
use vortex_compressor::scheme::SchemeId;
use vortex_error::VortexResult;
#[cfg(feature = "unstable_encodings")]
use vortex_fastlanes::Delta;
use vortex_fastlanes::RLE;
use vortex_fastlanes::RLEArrayExt;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::integer::RUN_THRESHOLD;
use crate::schemes::rle_ancestor_exclusions;
use crate::schemes::rle_descendant_exclusions;

/// RLE scheme for integer arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntRLEScheme;

/// Shared compression logic for RLE schemes.
pub(crate) fn rle_compress(
    scheme: &dyn Scheme,
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let rle_array = RLE::encode(data.array_as_primitive(), exec_ctx)?;

    let rle_values_primitive = rle_array
        .values()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?;
    let compressed_values = compressor.compress_child(
        &rle_values_primitive.into_array(),
        &compress_ctx,
        scheme.id(),
        0,
        exec_ctx,
    )?;

    // Delta is an unstable encoding, once we deem it stable we can switch over to this always.
    #[cfg(feature = "unstable_encodings")]
    let compressed_indices = {
        let rle_indices_primitive = rle_array
            .indices()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        try_compress_delta(
            compressor,
            &rle_indices_primitive.into_array(),
            &compress_ctx,
            scheme.id(),
            1,
            exec_ctx,
        )?
    };

    #[cfg(not(feature = "unstable_encodings"))]
    let compressed_indices = {
        let rle_indices_primitive = rle_array
            .indices()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?;
        compressor.compress_child(
            &rle_indices_primitive.into_array(),
            &compress_ctx,
            scheme.id(),
            1,
            exec_ctx,
        )?
    };

    let rle_offsets_primitive = rle_array
        .values_idx_offsets()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?
        .narrow(exec_ctx)?;
    let compressed_offsets = compressor.compress_child(
        &rle_offsets_primitive.into_array(),
        &compress_ctx,
        scheme.id(),
        2,
        exec_ctx,
    )?;

    // SAFETY: Recursive compression doesn't affect the invariants.
    unsafe {
        Ok(RLE::new_unchecked(
            compressed_values,
            compressed_indices,
            compressed_offsets,
            rle_array.offset(),
            rle_array.len(),
        )
        .into_array())
    }
}

#[cfg(feature = "unstable_encodings")]
pub(crate) fn try_compress_delta(
    compressor: &CascadingCompressor,
    child: &ArrayRef,
    parent_ctx: &CompressorContext,
    parent_id: SchemeId,
    child_index: usize,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let child_primitive = child.clone().execute::<PrimitiveArray>(exec_ctx)?;
    let (bases, deltas) = vortex_fastlanes::delta_compress(&child_primitive, exec_ctx)?;

    let compressed_bases = compressor.compress_child(
        &bases.into_array(),
        parent_ctx,
        parent_id,
        child_index,
        exec_ctx,
    )?;
    let compressed_deltas = compressor.compress_child(
        &deltas.into_array(),
        parent_ctx,
        parent_id,
        child_index,
        exec_ctx,
    )?;

    Delta::try_new(compressed_bases, compressed_deltas, 0, child.len()).map(IntoArray::into_array)
}

impl Scheme for IntRLEScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.rle"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    /// Children: values=0, indices=1, offsets=2.
    fn num_children(&self) -> usize {
        3
    }

    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        rle_descendant_exclusions()
    }

    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        rle_ancestor_exclusions()
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // RLE is only useful when we cascade it with another encoding.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        if data.integer_stats(exec_ctx).average_run_length() < RUN_THRESHOLD {
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
        rle_compress(self, compressor, data, compress_ctx, exec_ctx)
    }
}
