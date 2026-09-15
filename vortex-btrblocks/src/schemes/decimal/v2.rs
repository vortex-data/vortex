// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compression with lower parts for wide values.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::decimal::narrowed_decimal;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_decimal_byte_parts::DecimalBytePartsSlots;
use vortex_decimal_byte_parts::MAX_LOWER_PARTS;
use vortex_decimal_byte_parts::decimal_byte_parts_v2_id;
use vortex_decimal_byte_parts::split_decimal;
use vortex_error::VortexResult;

use super::DecimalScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Compression scheme for decimals with a signed most significant part and up to three lower parts.
///
/// Both byte-parts wire IDs must be permitted: wide values serialize under v2, while values that
/// narrow to a single part retain the frozen format. The compressor falls back to [`DecimalScheme`]
/// when only the frozen format is available.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DecimalSchemeV2;

impl Scheme for DecimalSchemeV2 {
    fn scheme_name(&self) -> &'static str {
        "vortex.decimal.byte_parts_v2"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        DecimalScheme.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![DecimalByteParts.id(), decimal_byte_parts_v2_id()]
    }

    fn predecessor(&self) -> Option<&'static dyn Scheme> {
        Some(&DecimalScheme)
    }

    /// Children: msp=0, then up to [`MAX_LOWER_PARTS`] lower parts.
    fn num_children(&self) -> usize {
        DecimalBytePartsSlots::FIXED_COUNT + MAX_LOWER_PARTS
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        DecimalScheme.expected_compression_ratio(data, compress_ctx, exec_ctx)
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
            .map(|d| d.into_array())
    }
}
