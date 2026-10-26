// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateArgs;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::fixed_size_bytes::FixedSizeBytes;

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
    partial.merge(
        args,
        extrema.map(|(min, max)| MinMaxResult {
            min: make_scalar(min),
            max: make_scalar(max),
        }),
    );
    Ok(())
}
