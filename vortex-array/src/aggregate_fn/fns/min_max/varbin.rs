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
    let mask = array
        .validity()?
        .execute_mask(array.len(), ctx)?
        .to_bit_buffer();
    let views = array.views();
    let buffers = array
        .data_buffers()
        .iter()
        .map(|b| b.as_host())
        .collect::<Vec<_>>();
    let minmax = views
        .iter()
        .zip(mask.iter())
        .filter(|(_, v)| *v)
        .map(|(view, _)| {
            if view.is_inlined() {
                view.as_inlined().value()
            } else {
                let view_ref = view.as_view();
                &buffers[view_ref.buffer_index as usize][view_ref.as_range()]
            }
        })
        .minmax();
    Ok(match minmax {
        itertools::MinMaxResult::NoElements => None,
        itertools::MinMaxResult::OneElement(value) => {
            let scalar = make_scalar(dtype, value);
            Some(MinMaxResult {
                min: scalar.clone(),
                max: scalar,
            })
        }
        itertools::MinMaxResult::MinMax(min, max) => Some(MinMaxResult {
            min: make_scalar(dtype, min),
            max: make_scalar(dtype, max),
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
