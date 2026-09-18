// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compression via byte-part decomposition.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::decimal::narrowed_decimal;
use vortex_array::dtype::DecimalType;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::DecimalBytePartsSlots;
use vortex_decimal_byte_parts::decimal_byte_parts_v1_id;
use vortex_decimal_byte_parts::decimal_byte_parts_v2_id;
use vortex_decimal_byte_parts::split_decimal;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Compression scheme for decimal arrays via byte-part decomposition.
///
/// Narrows the decimal to the smallest integer type and splits it into a signed most significant
/// part plus up to three unsigned 64-bit lower parts, each compressed as its own child. Values that
/// fit one signed part produce a single-part array under the frozen `vortex.decimal_byte_parts`
/// format. Wider values need lower parts, and so the `vortex.decimal_byte_parts.v2` format. They
/// are split only when the writer may emit that format, and stay canonical otherwise.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DecimalScheme {
    /// Whether the permitted serialized IDs include the format for lower parts.
    allow_v2: bool,
}

impl Default for DecimalScheme {
    fn default() -> Self {
        Self::new(None)
    }
}

impl DecimalScheme {
    /// Creates a decimal scheme using v2 for wide values when that format is permitted.
    ///
    /// `None` permits all serialized IDs. Availability is gated separately by
    /// [`Scheme::produced_encodings`].
    pub fn new(allowed_serialized_ids: Option<&HashSet<ArrayId>>) -> Self {
        Self {
            allow_v2: allowed_serialized_ids
                .is_none_or(|ids| ids.contains(&decimal_byte_parts_v2_id())),
        }
    }
}

impl Scheme for DecimalScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.decimal.byte_parts"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Decimal(_))
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        // Single-part arrays always serialize as v1, even when v2 is enabled.
        vec![decimal_byte_parts_v1_id()]
    }

    /// Children: msp=0, then up to three lower parts.
    fn num_children(&self) -> usize {
        4
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
        // Lower parts need the v2 format. Leave wide values canonical when the writer may not
        // emit it, so a frozen-format file never carries an array it cannot serialize.
        if !self.allow_v2 && matches!(decimal.values_type(), DecimalType::I128 | DecimalType::I256)
        {
            return Ok(decimal.into_array());
        }

        let parts = split_decimal(&decimal, exec_ctx)?;
        let msp = compressor.compress_child(
            &parts.msp,
            &compress_ctx,
            self.id(),
            DecimalBytePartsSlots::MSP,
            exec_ctx,
        )?;
        let lower_parts = parts
            .lower_parts
            .iter()
            .enumerate()
            .map(|(idx, part)| {
                compressor.compress_child(
                    part,
                    &compress_ctx,
                    self.id(),
                    DecimalBytePartsSlots::LOWER_PARTS_OFFSET + idx,
                    exec_ctx,
                )
            })
            .collect::<VortexResult<Vec<_>>>()?;

        DecimalByteParts::try_new_with_lower_parts(msp, lower_parts, decimal.decimal_dtype())
            .map(IntoArray::into_array)
    }
}
