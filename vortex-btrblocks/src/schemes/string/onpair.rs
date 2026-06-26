// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! OnPair short-string compression (dict-12).

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::scheme::SchemeId;
use vortex_error::VortexResult;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArrayExt;
use vortex_onpair::OnPairArraySlotsExt;
use vortex_onpair::onpair_compress;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::integer::try_compress_delta;

/// OnPair short-string compression (dict-12).
///
/// A default string-fragmentation scheme (alongside [`super::FSSTScheme`]) —
/// targets large columns of short-to-medium strings with high lexical
/// overlap, like URLs or log lines. Uses a learned dictionary of frequent
/// adjacent substrings (built by the OnPair trainer at compress time) and
/// 12-bit token codes stored as a u16 child, with offsets /
/// uncompressed-lengths flowing through the cascading compressor like any
/// other primitive children.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct OnPairScheme;

impl Scheme for OnPairScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.onpair"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    /// 4 primitive slot children flow through the cascading compressor:
    /// `dict_offsets` (u32 → typically `FoR`/`BitPacked`), `codes` (u16 →
    /// `FastLanes::BitPacked` to exactly `bits` = 12 by default),
    /// `codes_offsets` (u32 → `FoR`), `uncompressed_lengths` (i32 → narrow
    /// + `FoR`). Validity stays untouched.
    fn num_children(&self) -> usize {
        4
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let utf8 = data.array_as_varbinview().into_owned();
        let onpair_array = onpair_compress(utf8.as_array(), DEFAULT_DICT12_CONFIG, exec_ctx)?;

        let dict_offsets = compress_offsets_child(
            compressor,
            onpair_array.dict_offsets(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;
        let codes = compress_primitive_child(
            compressor,
            onpair_array.codes(),
            &compress_ctx,
            self.id(),
            1,
            exec_ctx,
        )?;
        let codes_offsets = compress_offsets_child(
            compressor,
            onpair_array.codes_offsets(),
            &compress_ctx,
            self.id(),
            2,
            exec_ctx,
        )?;
        let uncompressed_lengths = compress_primitive_child(
            compressor,
            onpair_array.uncompressed_lengths(),
            &compress_ctx,
            self.id(),
            3,
            exec_ctx,
        )?;

        Ok(OnPair::try_new(
            onpair_array.dtype().clone(),
            onpair_array.dict_bytes_handle().clone(),
            dict_offsets,
            codes,
            codes_offsets,
            uncompressed_lengths,
            onpair_array.array_validity(),
            onpair_array.bits(),
        )?
        .into_array())
    }
}

/// Narrow a primitive child to its tightest int type, then forward it to
/// the cascading compressor.
fn compress_primitive_child(
    compressor: &CascadingCompressor,
    child: &ArrayRef,
    compress_ctx: &CompressorContext,
    scheme_id: SchemeId,
    child_idx: usize,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let narrowed = child
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?
        .narrow(exec_ctx)?
        .into_array();
    compressor.compress_child(&narrowed, compress_ctx, scheme_id, child_idx, exec_ctx)
}

/// Minimum child length before delta is even attempted. Delta carries fixed
/// overhead (a separate `bases` array plus FastLanes' 1024-element lane
/// packing), so on short children it can only lose.
const OFFSETS_DELTA_MIN_LEN: usize = 2048;

/// Compress a monotonic offsets child. For children of at least
/// [`OFFSETS_DELTA_MIN_LEN`] it tries both the normal cascading path and a
/// delta path and keeps whichever produces fewer bytes; shorter children
/// skip delta entirely. `dict_offsets` and `codes_offsets` are cumulative
/// (monotonic), so delta (per-entry deltas) usually packs much tighter than
/// FoR+bitpacking over the full range.
fn compress_offsets_child(
    compressor: &CascadingCompressor,
    child: &ArrayRef,
    compress_ctx: &CompressorContext,
    scheme_id: SchemeId,
    child_idx: usize,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let narrowed = child
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?
        .narrow(exec_ctx)?
        .into_array();
    let plain =
        compressor.compress_child(&narrowed, compress_ctx, scheme_id, child_idx, exec_ctx)?;
    if narrowed.len() < OFFSETS_DELTA_MIN_LEN {
        return Ok(plain);
    }
    let delta = try_compress_delta(
        compressor,
        &narrowed,
        compress_ctx,
        scheme_id,
        child_idx,
        exec_ctx,
    )?;
    if delta.nbytes() < plain.nbytes() {
        Ok(delta)
    } else {
        Ok(plain)
    }
}
