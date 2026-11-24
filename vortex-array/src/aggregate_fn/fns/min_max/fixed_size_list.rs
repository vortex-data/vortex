// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::fixed_size_bytes::FixedSizeBytes;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::dtype::Nullability::NonNullable;
use crate::scalar::Scalar;

/// Accumulate the byte-wise extrema of a fixed-size list of bytes.
pub(super) fn accumulate_fixed_size_bytes(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    array: &FixedSizeListArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let mask = array
        .as_ref()
        .validity()?
        .execute_mask(array.len(), ctx)?
        .to_bit_buffer();
    let bytes = FixedSizeBytes::try_new(array, ctx)?;

    let (element_dtype, ..) = array.dtype_parts();
    let make_scalar = |value: &[u8]| {
        Scalar::fixed_size_list(
            element_dtype.clone(),
            value
                .iter()
                .map(|&byte| Scalar::primitive(byte, NonNullable))
                .collect(),
            NonNullable,
        )
    };
    partial.merge(args, byte_extrema(&bytes, &mask, make_scalar))
}

/// Accumulate the byte-wise extrema of a fixed-size binary array.
pub(super) fn accumulate_fixed_size_binary(
    args: AggregateArgs<'_, NumericalAggregateOpts>,
    partial: &mut MinMaxPartial,
    array: &FixedSizeBinaryArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let mask = array
        .validity()?
        .execute_mask(array.len(), ctx)?
        .to_bit_buffer();
    let bytes = FixedSizeBytes::from_fixed_size_binary(array);

    let make_scalar = |value: &[u8]| Scalar::fixed_size_binary(value.to_vec(), NonNullable);
    partial.merge(args, byte_extrema(&bytes, &mask, make_scalar))
}

/// Returns the smallest and largest valid values by byte, as scalars made by `make_scalar`.
fn byte_extrema(
    bytes: &FixedSizeBytes,
    mask: &BitBuffer,
    make_scalar: impl Fn(&[u8]) -> Scalar,
) -> Option<MinMaxResult> {
    let extrema = bytes
        .values()
        .zip(mask.iter())
        .filter_map(|(value, valid)| valid.then_some(value))
        .fold(None, |extrema: Option<(&[u8], &[u8])>, value| {
            Some(match extrema {
                None => (value, value),
                Some((min, max)) => (min.min(value), max.max(value)),
            })
        });
    extrema.map(|(min, max)| MinMaxResult {
        min: make_scalar(min),
        max: make_scalar(max),
    })
}
