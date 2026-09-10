// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! OnPair short-string compression (dict-12).

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::SchemeId;
use vortex_error::VortexResult;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArrayExt;
use vortex_onpair::OnPairArraySlotsExt;
use vortex_onpair::OnPairIndexChildren;
use vortex_onpair::OnPairIndexSet;
use vortex_onpair::OnPairSlots;
use vortex_onpair::build_token_frequency_index;
use vortex_onpair::onpair_compress;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

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
pub struct OnPairScheme {
    indexes: OnPairIndexSet,
}

impl OnPairScheme {
    /// Create an OnPair scheme without auxiliary indexes.
    pub const fn new() -> Self {
        Self {
            indexes: OnPairIndexSet::empty(),
        }
    }

    /// Include the token-frequency index used by substring prefiltering.
    ///
    /// The index is built only after OnPair wins full-column scheme selection,
    /// so it does not affect the sampling estimate.
    ///
    /// ```
    /// use vortex_btrblocks::schemes::string::OnPairScheme;
    /// use vortex_btrblocks::{BtrBlocksCompressorBuilder, SchemeExt};
    ///
    /// const FREQUENCY_INDEXED_ONPAIR: OnPairScheme =
    ///     OnPairScheme::new().with_token_frequency_index();
    ///
    /// let compressor = BtrBlocksCompressorBuilder::default()
    ///     .exclude_schemes([OnPairScheme::new().id()])
    ///     .with_new_scheme(&FREQUENCY_INDEXED_ONPAIR)
    ///     .build();
    /// # let _ = compressor;
    /// ```
    pub const fn with_token_frequency_index(mut self) -> Self {
        self.indexes = self.indexes.with_token_frequency();
        self
    }
}

impl Default for OnPairScheme {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheme for OnPairScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.string.onpair"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_utf8()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![OnPair.id()]
    }

    /// The required primitive slot children flow through the cascading compressor:
    /// `dict_offsets` (u32 → typically `FoR`/`BitPacked`), `codes` (u16 →
    /// usually `FastLanes::BitPacked` after scheme selection),
    /// `codes_offsets` (u32 → `FoR`), `uncompressed_lengths` (i32 → narrow
    /// + `FoR`). Validity stays untouched.
    fn num_children(&self) -> usize {
        OnPairSlots::VALIDITY + usize::from(self.indexes.has_token_frequency())
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
        compress_onpair_scheme(
            compressor,
            data,
            compress_ctx,
            exec_ctx,
            self.id(),
            self.indexes,
        )
    }
}

fn compress_onpair_scheme(
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    compress_ctx: CompressorContext,
    exec_ctx: &mut ExecutionCtx,
    scheme_id: SchemeId,
    indexes: OnPairIndexSet,
) -> VortexResult<ArrayRef> {
    let utf8 = data.array_as_varbinview().into_owned();
    let encoded = onpair_compress(utf8.as_array(), DEFAULT_CONFIG, exec_ctx)?;
    let mut onpair_array = match encoded.try_downcast::<OnPair>() {
        Ok(array) => array,
        Err(array) => return Ok(array),
    };
    if !compress_ctx.is_sample() && indexes.has_token_frequency() {
        onpair_array = build_token_frequency_index(onpair_array, exec_ctx)?;
    }

    let dict_offsets = compress_primitive_child(
        compressor,
        onpair_array.dict_offsets(),
        &compress_ctx,
        scheme_id,
        0,
        exec_ctx,
    )?;
    let codes = compress_primitive_child(
        compressor,
        onpair_array.codes(),
        &compress_ctx,
        scheme_id,
        1,
        exec_ctx,
    )?;
    let codes_offsets = compress_primitive_child(
        compressor,
        onpair_array.codes_offsets(),
        &compress_ctx,
        scheme_id,
        2,
        exec_ctx,
    )?;
    let uncompressed_lengths = compress_primitive_child(
        compressor,
        onpair_array.uncompressed_lengths(),
        &compress_ctx,
        scheme_id,
        3,
        exec_ctx,
    )?;

    let index_children = onpair_array
        .token_frequency_index_child()
        .map(|child| {
            compressor.compress_child(
                child,
                &compress_ctx,
                scheme_id,
                OnPairSlots::VALIDITY,
                exec_ctx,
            )
        })
        .transpose()?
        .map(|child| OnPairIndexChildren::default().with_token_frequency(child))
        .unwrap_or_default();

    Ok(OnPair::try_new_with_data(
        onpair_array.dtype().clone(),
        onpair_array.data().clone(),
        dict_offsets,
        codes,
        codes_offsets,
        uncompressed_lengths,
        onpair_array.array_validity(),
        index_children,
    )?
    .into_array())
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

// TODO(joe): re-apply Delta to the monotone `dict_offsets` / `codes_offsets` children once the
// compressor can say whether a scheme is eligible for a given child; applying it by hand
// bypassed the exclusion rules.
