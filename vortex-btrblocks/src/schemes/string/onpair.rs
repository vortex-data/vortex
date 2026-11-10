// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! OnPair short-string compression (dict-12 by default).

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
use vortex_onpair::Config;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::MaxDictBits;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArrayExt;
use vortex_onpair::OnPairArraySlotsExt;
use vortex_onpair::onpair_compress;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// OnPair short-string compression (dict-12 by default).
///
/// A default string-fragmentation scheme (alongside [`super::FSSTScheme`]) —
/// targets large columns of short-to-medium strings with high lexical
/// overlap, like URLs or log lines. Uses a learned dictionary of frequent
/// adjacent substrings (built by the OnPair trainer at compress time) and
/// token codes of up to `max_dict_bits` bits stored as a u16 child, with
/// offsets / uncompressed-lengths flowing through the cascading compressor
/// like any other primitive children.
///
/// Every dictionary budget shares one [`SchemeId`], so a scheme built with
/// [`OnPairScheme::new`] replaces the default one rather than competing with it.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct OnPairScheme {
    max_dict_bits: MaxDictBits,
}

impl OnPairScheme {
    /// Creates an OnPair scheme whose dictionary holds up to `2^max_dict_bits` tokens, keeping
    /// every other [`DEFAULT_CONFIG`] setting. [`OnPairScheme::default`] uses 12 bits.
    pub const fn new(max_dict_bits: MaxDictBits) -> Self {
        Self { max_dict_bits }
    }

    /// The dictionary budget this scheme trains with.
    pub const fn max_dict_bits(&self) -> MaxDictBits {
        self.max_dict_bits
    }

    fn config(&self) -> Config {
        Config {
            max_dict_bits: self.max_dict_bits,
            ..DEFAULT_CONFIG
        }
    }
}

impl Default for OnPairScheme {
    fn default() -> Self {
        Self::new(DEFAULT_CONFIG.max_dict_bits)
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

    /// 4 primitive slot children flow through the cascading compressor:
    /// `dict_offsets` (u32 → typically `FoR`/`BitPacked`), `codes` (u16 →
    /// usually `FastLanes::BitPacked` after scheme selection),
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
        let encoded = onpair_compress(utf8.as_array(), self.config(), exec_ctx)?;
        let Some(onpair_array) = encoded.as_opt::<OnPair>() else {
            return Ok(encoded);
        };

        let dict_offsets = compress_primitive_child(
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
        let codes_offsets = compress_primitive_child(
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

        Ok(OnPair::try_new_with_data(
            onpair_array.dtype().clone(),
            onpair_array.data().clone(),
            dict_offsets,
            codes,
            codes_offsets,
            uncompressed_lengths,
            onpair_array.array_validity(),
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

// TODO(joe): re-apply Delta to the monotone `dict_offsets` / `codes_offsets` children once the
// compressor can say whether a scheme is eligible for a given child; applying it by hand
// bypassed the exclusion rules.
