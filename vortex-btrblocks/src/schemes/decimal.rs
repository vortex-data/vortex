// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compression scheme using byte-part decomposition.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
#[expect(deprecated)]
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::decimal::narrowed_decimal;
use vortex_array::dtype::DecimalType;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_decimal_byte_parts::DecimalByteParts;
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DecimalScheme;

impl Scheme for DecimalScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.decimal.byte_parts"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Decimal(_))
    }

    /// Children: primitive=0.
    fn num_children(&self) -> usize {
        1
    }

    fn expected_compression_ratio(
        &self,
        _data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        // Decimal compression is almost always beneficial (narrowing + primitive compression).
        CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        // TODO(joe): add support splitting i128/256 buffers into chunks of primitive values
        // for compression. 2 for i128 and 4 for i256.
        #[expect(deprecated)]
        let decimal = data.array().clone().to_decimal();
        let decimal = narrowed_decimal(decimal);
        let validity = decimal.validity()?;
        let prim = match decimal.values_type() {
            DecimalType::I8 => PrimitiveArray::new(decimal.buffer::<i8>(), validity),
            DecimalType::I16 => PrimitiveArray::new(decimal.buffer::<i16>(), validity),
            DecimalType::I32 => PrimitiveArray::new(decimal.buffer::<i32>(), validity),
            DecimalType::I64 => PrimitiveArray::new(decimal.buffer::<i64>(), validity),
            _ => return Ok(decimal.into_array()),
        };

        let compressed = compressor.compress_child(&prim.into_array(), &ctx, self.id(), 0)?;

        DecimalByteParts::try_new(compressed, decimal.decimal_dtype()).map(|d| d.into_array())
    }
}
