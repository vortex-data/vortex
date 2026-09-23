// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compression scheme using byte-part decomposition.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::decimal::narrowed_decimal;
use vortex_array::dtype::DecimalType;
use vortex_compressor::scheme::AllowedSerializedIds;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::DecimalBytePartsSlots;
use vortex_decimal_byte_parts::decimal_byte_parts_v1_id;
use vortex_decimal_byte_parts::decimal_byte_parts_v2_id;
use vortex_decimal_byte_parts::split_decimal;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum DecimalSchemeMode {
    V1,
    V2,
}

static DECIMAL_V2: DecimalScheme = DecimalScheme::v2();

/// Compression scheme for decimal arrays via byte-part decomposition.
///
/// Narrows the decimal to the smallest integer type and compresses its byte parts independently.
/// The v1 mode leaves values wider than `i64` canonical; v2 splits them into a signed most
/// significant part and up to three unsigned lower parts. Single-part arrays serialize as v1
/// in either mode, while arrays with lower parts serialize as v2.
///
/// The default uses v1. Permitting both decimal IDs lets the builder upgrade v1 to v2.
/// A v2 scheme is filtered out if either ID is forbidden, including under the CUDA preset.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DecimalScheme {
    mode: DecimalSchemeMode,
}

impl DecimalScheme {
    /// Creates a decimal scheme configured for v1, disallowing splitting of wide decimals.
    ///
    /// Values that remain wider than `i64` after narrowing stay canonical.
    /// The builder may upgrade to v2 if its permissions include both serialized IDs.
    pub const fn v1() -> Self {
        Self {
            mode: DecimalSchemeMode::V1,
        }
    }

    /// Creates a decimal scheme configured for v2, allowing splitting of wide decimals.
    /// The builder filters this scheme out if either decimal serialized ID is forbidden.
    pub const fn v2() -> Self {
        Self {
            mode: DecimalSchemeMode::V2,
        }
    }
}

impl Default for DecimalScheme {
    fn default() -> Self {
        Self::v1()
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
        match self.mode {
            DecimalSchemeMode::V1 => vec![decimal_byte_parts_v1_id()],
            DecimalSchemeMode::V2 => {
                vec![decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]
            }
        }
    }

    fn try_upgrade(&self, allowed_serialized_ids: &AllowedSerializedIds) -> Option<&dyn Scheme> {
        if self.mode == DecimalSchemeMode::V1
            && allowed_serialized_ids.contains(&decimal_byte_parts_v1_id())
            && allowed_serialized_ids.contains(&decimal_byte_parts_v2_id())
        {
            Some(&DECIMAL_V2)
        } else {
            None
        }
    }

    /// Children: msp=0, then up to three lower parts in v2 mode.
    fn num_children(&self) -> usize {
        match self.mode {
            DecimalSchemeMode::V1 => 1,
            DecimalSchemeMode::V2 => 4,
        }
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
        if self.mode == DecimalSchemeMode::V1
            && matches!(decimal.values_type(), DecimalType::I128 | DecimalType::I256)
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
