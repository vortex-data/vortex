// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compression scheme using byte-part decomposition.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::decimal::narrowed_decimal;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::DecimalBytePartsSlots;
use vortex_decimal_byte_parts::split_decimal;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Compression scheme for decimal arrays via byte-part decomposition.
///
/// Narrows the decimal to the smallest integer type, compresses the underlying primitive, and wraps
/// the result in a `DecimalBytePartsArray`.
///
/// Only decimals that fit a single signed part are compressed. Anything still wider than 64
/// bits after narrowing would need lower parts, which cannot be serialized, so those are left
/// as the canonical decimal.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DecimalScheme;

impl Scheme for DecimalScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.decimal.byte_parts"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Decimal(_))
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![DecimalByteParts.id()]
    }

    /// Children: msp=0. This scheme never emits lower parts.
    fn num_children(&self) -> usize {
        DecimalBytePartsSlots::FIXED_COUNT
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // Decimal compression is almost always beneficial (narrowing + primitive compression).
        CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let decimal = data.array().clone().execute::<DecimalArray>(exec_ctx)?;
        let decimal = narrowed_decimal(decimal);
        let parts = split_decimal(&decimal)?;

        // A value too wide for one signed part splits into lower parts, which serialize under
        // the v2 format id — one this scheme does not declare in `produced_encodings`, so a
        // writer restricted to its editions could refuse it. Leave it as the canonical decimal
        // rather than build something outside the scheme's declared output.
        if !parts.lower_parts.is_empty() {
            return Ok(decimal.into_array());
        }

        let msp = compressor.compress_child(
            &parts.msp,
            &compress_ctx,
            self.id(),
            DecimalBytePartsSlots::MSP,
            exec_ctx,
        )?;
        DecimalByteParts::try_new(msp, decimal.decimal_dtype()).map(|d| d.into_array())
    }
}

#[cfg(test)]
mod tests;
