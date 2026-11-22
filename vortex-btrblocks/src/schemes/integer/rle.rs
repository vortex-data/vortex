// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run-length integer encoding and shared RLE compression helpers.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::RLE;
use vortex_fastlanes::RLEArrayExt;
use vortex_fastlanes::RLEArraySlotsExt;

use super::RUN_LENGTH_THRESHOLD;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::Child;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;
use crate::schemes::rle_ancestor_exclusions;
use crate::schemes::rle_descendant_exclusions;

/// RLE scheme for integer arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntRLEScheme;

/// Shared encoding logic for RLE schemes.
///
/// Children: values=0, indices=1, offsets=2. The indices carry the validity.
pub(crate) fn rle_encode(
    scheme: &dyn Scheme,
    data: &ArrayAndStats,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Encoded> {
    let rle_array = RLE::encode(data.array_as_primitive(), exec_ctx)?;

    let values = rle_array
        .values()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?;

    // TODO(joe): re-apply Delta to these monotone indices once the compressor can say whether a
    // scheme is eligible for a given child; applying it by hand bypassed the exclusion rules.
    let indices = rle_array
        .indices()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?
        .narrow(exec_ctx)?;

    let offsets = rle_array
        .values_idx_offsets()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?
        .narrow(exec_ctx)?;

    let offset = rle_array.offset();
    let len = rle_array.len();
    Ok(Encoded::new(
        vec![
            Child::new(scheme, 0, values.into_array()),
            Child::new(scheme, 1, indices.into_array()),
            Child::new(scheme, 2, offsets.into_array()),
        ],
        Vec::new(),
        move |children, _| {
            let [values, indices, offsets] = children.compressed()?;
            // SAFETY: Recursive compression doesn't affect the invariants.
            Ok(unsafe { RLE::new_unchecked(values, indices, offsets, offset, len) }.into_array())
        },
    ))
}

impl Scheme for IntRLEScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.rle"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![RLE.id()]
    }

    fn top_level_encodings(&self) -> Vec<ArrayId> {
        vec![RLE.id()]
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
        if data.integer_stats(exec_ctx).average_run_length() < RUN_LENGTH_THRESHOLD {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    /// Children: values=0, indices=1, offsets=2. The indices carry the validity.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        rle_encode(self, data, exec_ctx)
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
