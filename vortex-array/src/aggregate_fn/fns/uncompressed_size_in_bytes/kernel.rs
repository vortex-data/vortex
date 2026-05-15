// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::UncompressedSizeInBytes;
use super::checked_len_mul;
use super::packed_bit_buffer_size_in_bytes;
use super::validity_uncompressed_size_in_bytes;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::scalar::Scalar;

/// Computes [`UncompressedSizeInBytes`] for fixed-width logical dtypes without decoding values.
///
/// This kernel is intended for physical encodings whose logical type is `Bool`, `Primitive`,
/// `Decimal`, or an extension over one of those types. Variable-width and nested dtypes return
/// `None` so another encoding-specific kernel or the canonical fallback can handle them.
#[derive(Debug)]
pub struct FixedWidthUncompressedSizeInBytesKernel;

impl DynAggregateKernel for FixedWidthUncompressedSizeInBytesKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<UncompressedSizeInBytes>() {
            return Ok(None);
        }

        Ok(fixed_width_uncompressed_size_in_bytes(batch, ctx)?.map(Scalar::from))
    }
}

pub(crate) fn fixed_width_uncompressed_size_in_bytes(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<u64>> {
    let Some(value_size) = fixed_width_value_size(array.dtype(), array.len())? else {
        return Ok(None);
    };

    if value_size.include_validity {
        let validity_size =
            validity_uncompressed_size_in_bytes(array.validity()?.execute_mask(array.len(), ctx)?)?;
        value_size
            .size
            .checked_add(validity_size)
            .map(Some)
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
    } else {
        Ok(Some(value_size.size))
    }
}

struct FixedWidthValueSize {
    size: u64,
    include_validity: bool,
}

fn fixed_width_value_size(dtype: &DType, len: usize) -> VortexResult<Option<FixedWidthValueSize>> {
    let fixed = match dtype {
        DType::Null => FixedWidthValueSize {
            size: 0,
            include_validity: false,
        },
        DType::Bool(_) => FixedWidthValueSize {
            size: packed_bit_buffer_size_in_bytes(len)?,
            include_validity: true,
        },
        DType::Primitive(ptype, _) => FixedWidthValueSize {
            size: checked_len_mul(len, ptype.byte_width(), "primitive")?,
            include_validity: true,
        },
        DType::Decimal(decimal_type, _) => FixedWidthValueSize {
            size: checked_len_mul(
                len,
                DecimalType::smallest_decimal_value_type(decimal_type).byte_width(),
                "decimal",
            )?,
            include_validity: true,
        },
        DType::Extension(ext_dtype) => {
            return fixed_width_value_size(ext_dtype.storage_dtype(), len);
        }
        DType::Utf8(_)
        | DType::Binary(_)
        | DType::List(..)
        | DType::FixedSizeList(..)
        | DType::Struct(..)
        | DType::Union(_)
        | DType::Variant(_) => return Ok(None),
    };

    Ok(Some(fixed))
}
