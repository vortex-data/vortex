// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal compression scheme using byte-part decomposition.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::decimal::narrowed_decimal;
use vortex_array::dtype::DecimalType;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
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

    /// Children: the most significant part (index 0) plus up to three unsigned lower parts for
    /// `i128`/`i256` decimals.
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
        let validity = decimal.validity()?;
        let decimal_dtype = decimal.decimal_dtype();

        // Values that fit in a signed 64-bit (or narrower) integer become a single msp part. Wider
        // decimals are split into a signed msp plus unsigned 64-bit lower parts, each compressed
        // independently.
        let (msp, lower_parts) = match decimal.values_type() {
            DecimalType::I8 => (
                PrimitiveArray::new(decimal.buffer::<i8>(), validity),
                Vec::new(),
            ),
            DecimalType::I16 => (
                PrimitiveArray::new(decimal.buffer::<i16>(), validity),
                Vec::new(),
            ),
            DecimalType::I32 => (
                PrimitiveArray::new(decimal.buffer::<i32>(), validity),
                Vec::new(),
            ),
            DecimalType::I64 => (
                PrimitiveArray::new(decimal.buffer::<i64>(), validity),
                Vec::new(),
            ),
            DecimalType::I128 => split_i128(&decimal, validity),
            DecimalType::I256 => split_i256(&decimal, validity),
        };

        let compressed_msp =
            compressor.compress_child(&msp.into_array(), &compress_ctx, self.id(), 0, exec_ctx)?;

        let compressed_lower = lower_parts
            .into_iter()
            .enumerate()
            .map(|(idx, part)| {
                compressor.compress_child(
                    &part.into_array(),
                    &compress_ctx,
                    self.id(),
                    idx + 1,
                    exec_ctx,
                )
            })
            .collect::<VortexResult<Vec<_>>>()?;

        DecimalByteParts::try_new_parts(compressed_msp, compressed_lower, decimal_dtype)
            .map(|d| d.into_array())
    }
}

/// Splits an `i128`-backed decimal into a signed high 64-bit msp and one unsigned low 64-bit part.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "extracting 64-bit chunks of a wider integer intentionally truncates"
)]
fn split_i128(decimal: &DecimalArray, validity: Validity) -> (PrimitiveArray, Vec<PrimitiveArray>) {
    let buffer = decimal.buffer::<i128>();
    let mut msp = BufferMut::<i64>::with_capacity(buffer.len());
    let mut low = BufferMut::<u64>::with_capacity(buffer.len());
    for value in buffer.iter() {
        msp.push((*value >> 64) as i64);
        low.push(*value as u64);
    }
    (
        PrimitiveArray::new(msp.freeze(), validity),
        vec![PrimitiveArray::new(low.freeze(), Validity::NonNullable)],
    )
}

/// Splits an `i256`-backed decimal into a signed high 64-bit msp and three unsigned 64-bit parts,
/// in most-significant-first order.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "extracting 64-bit chunks of a wider integer intentionally truncates"
)]
fn split_i256(decimal: &DecimalArray, validity: Validity) -> (PrimitiveArray, Vec<PrimitiveArray>) {
    let buffer = decimal.buffer::<vortex_array::dtype::i256>();
    let mut msp = BufferMut::<i64>::with_capacity(buffer.len());
    let mut p0 = BufferMut::<u64>::with_capacity(buffer.len());
    let mut p1 = BufferMut::<u64>::with_capacity(buffer.len());
    let mut p2 = BufferMut::<u64>::with_capacity(buffer.len());
    for value in buffer.iter() {
        let (lower, upper) = value.to_parts();
        msp.push((upper >> 64) as i64);
        p0.push(upper as u64);
        p1.push((lower >> 64) as u64);
        p2.push(lower as u64);
    }
    (
        PrimitiveArray::new(msp.freeze(), validity),
        vec![
            PrimitiveArray::new(p0.freeze(), Validity::NonNullable),
            PrimitiveArray::new(p1.freeze(), Validity::NonNullable),
            PrimitiveArray::new(p2.freeze(), Validity::NonNullable),
        ],
    )
}
