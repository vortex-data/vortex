// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ExecutionCtx;
use crate::arrays::VarBinViewArray;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::scalar::Scalar;

pub(super) fn accumulate_varbinview(
    partial: &mut MinMaxPartial,
    array: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    partial.merge(varbin_compute_min_max(array, array.dtype(), ctx)?);
    Ok(())
}

fn varbin_compute_min_max(
    array: &VarBinViewArray,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<MinMaxResult>> {
    let mask = array.validity()?.execute_mask(array.len(), ctx)?;
    let minmax = (0..array.len())
        .filter(|&i| mask.value(i))
        .map(|i| array.bytes_at(i))
        .minmax();
    Ok(match minmax {
        itertools::MinMaxResult::NoElements => None,
        itertools::MinMaxResult::OneElement(value) => {
            let scalar = make_scalar(dtype, value.as_slice());
            Some(MinMaxResult {
                min: scalar.clone(),
                max: scalar,
            })
        }
        itertools::MinMaxResult::MinMax(min, max) => Some(MinMaxResult {
            min: make_scalar(dtype, min.as_slice()),
            max: make_scalar(dtype, max.as_slice()),
        }),
    })
}

fn make_scalar(dtype: &DType, value: &[u8]) -> Scalar {
    match dtype {
        DType::Binary(_) => Scalar::binary(value.to_vec(), NonNullable),
        DType::Utf8(_) => {
            // SAFETY: VarBin arrays always validate their data against their dtype.
            let value = unsafe { str::from_utf8_unchecked(value) };
            Scalar::utf8(value, NonNullable)
        }
        _ => vortex_panic!("cannot make Scalar from bytes with dtype {dtype}"),
    }
}
