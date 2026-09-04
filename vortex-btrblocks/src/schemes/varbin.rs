// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Offset-based storage for variable-length arrays.
//!
//! Canonical binary and UTF-8 arrays are [`VarBinViewArray`], which spends a fixed 16 bytes per
//! element on an opaque views buffer that no scheme can compress. Re-encoding as [`VarBinArray`]
//! replaces that buffer with an offsets child array, which the cascading compressor can then
//! compress with the ordinary integer schemes. For fixed-width values the offsets are a
//! constant-stride sequence and collapse to nothing.
//!
//! The logic is identical for both logical types, so a single implementation is registered twice:
//! once as [`VarBinScheme::BINARY`] and once as [`VarBinScheme::UTF8`].
//!
//! [`VarBinViewArray`]: vortex_array::arrays::VarBinViewArray

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::varbin::VarBinArraySlotsExt;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::SchemeExt;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

/// Offset-based (rather than view-based) storage for variable-length arrays.
///
/// Use the [`BINARY`](Self::BINARY) and [`UTF8`](Self::UTF8) constants; each selects the logical
/// type the scheme applies to, and both share this implementation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct VarBinScheme {
    utf8: bool,
}

impl VarBinScheme {
    /// The scheme instance for binary arrays.
    pub const BINARY: Self = Self { utf8: false };

    /// The scheme instance for UTF-8 arrays.
    pub const UTF8: Self = Self { utf8: true };
}

impl Scheme for VarBinScheme {
    fn scheme_name(&self) -> &'static str {
        if self.utf8 {
            "vortex.string.varbin"
        } else {
            "vortex.binary.varbin"
        }
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        if self.utf8 {
            canonical.dtype().is_utf8()
        } else {
            canonical.dtype().is_binary()
        }
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![VarBin.id()]
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
        let view = data.array_as_varbinview().into_owned();
        let len = view.len();
        // Materialize validity once; a per-element accessor here would be quadratic-ish in a
        // loop this hot.
        let mask = view.validity()?.execute_mask(len, exec_ctx)?;

        let varbin = VarBinArray::from_iter(
            (0..len).map(|i| mask.value(i).then(|| view.bytes_at(i).as_slice().to_vec())),
            view.dtype().clone(),
        );

        let offsets = varbin
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?
            .narrow(exec_ctx)?
            .into_array();
        let compressed_offsets =
            compressor.compress_child(&offsets, &compress_ctx, self.id(), 0, exec_ctx)?;

        Ok(VarBinArray::try_new(
            compressed_offsets,
            varbin.bytes().clone(),
            varbin.dtype().clone(),
            varbin.validity()?,
        )?
        .into_array())
    }
}
