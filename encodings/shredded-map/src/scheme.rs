// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressor schemes for maps with UTF-8 keys.
//!
//! Register them with `BtrBlocksCompressorBuilder::with_new_scheme`. The compressor tries every
//! eligible map scheme next to compressing the map's children as they are, and keeps the smallest.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Dict;
use vortex_array::dtype::DType;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::CompressorContext;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::scheme::SchemeExt;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexResult;

use crate::ShredOptions;
use crate::ShreddedMap;
use crate::compress_encoded;
use crate::encode;
use crate::keyset::KeySetMap;
use crate::keyset::KeySetMapArraySlotsExt;
use crate::keyset::KeySetOptions;
use crate::keyset::keyset_encode;

fn utf8_keys(canonical: &Canonical) -> bool {
    matches!(canonical, Canonical::Map(m) if matches!(m.dtype(), DType::Map(t, _) if matches!(t.key_dtype(), DType::Utf8(_))))
}

/// Stores each distinct key set once, see [`KeySetMap`].
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct KeySetMapScheme {
    /// Also let rows equal to their predecessor share its values.
    pub dedup_values: bool,
}

/// [`KeySetMapScheme`] storing the values of every row.
pub static KEYSET_SCHEME: KeySetMapScheme = KeySetMapScheme {
    dedup_values: false,
};

/// [`KeySetMapScheme`] sharing the values of repeated rows.
pub static KEYSET_ROWS_SCHEME: KeySetMapScheme = KeySetMapScheme { dedup_values: true };

impl Scheme for KeySetMapScheme {
    fn scheme_name(&self) -> &'static str {
        if self.dedup_values {
            "vortex.map.keyset_rows"
        } else {
            "vortex.map.keyset"
        }
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        utf8_keys(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![KeySetMap.id()]
    }

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
        let options = KeySetOptions {
            dedup_values: self.dedup_values,
        };
        let encoded = keyset_encode(data.array(), options, exec_ctx)?;
        let child = |array: &ArrayRef, index: usize, ctx: &mut ExecutionCtx| {
            compressor.compress_child(array, &compress_ctx, self.id(), index, ctx)
        };
        Ok(KeySetMap::try_new(
            encoded.dtype().clone(),
            child(encoded.keyset_ids(), 0, exec_ctx)?,
            child(encoded.keysets(), 1, exec_ctx)?,
            child(encoded.value_offsets(), 2, exec_ctx)?,
            child(encoded.values(), 3, exec_ctx)?,
        )?
        .into_array())
    }
}

/// Shreds frequent keys into columns after deduplicating repeated rows, see [`encode`].
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ShreddedMapScheme;

/// The [`ShreddedMapScheme`].
pub static SHREDDED_SCHEME: ShreddedMapScheme = ShreddedMapScheme;

impl Scheme for ShreddedMapScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.map.shredded"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        utf8_keys(canonical)
            && matches!(canonical, Canonical::Map(m) if matches!(m.dtype(), DType::Map(t, _) if t.keys_sorted()))
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![ShreddedMap.id(), Dict.id()]
    }

    fn num_children(&self) -> usize {
        1
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
        let encoded = encode(data.array(), &ShredOptions::default(), exec_ctx)?;
        // Every child shares one cascade slot, which keeps this scheme off its own residual.
        compress_encoded(&encoded, |child| {
            compressor.compress_child(child, &compress_ctx, self.id(), 0, exec_ctx)
        })
    }
}
