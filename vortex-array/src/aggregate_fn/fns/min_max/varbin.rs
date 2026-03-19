// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::accessor::ArrayAccessor;
use crate::arrays::VarBinViewArray;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::scalar::Scalar;

pub(super) fn accumulate_varbinview(
    partial: &mut MinMaxPartial,
    array: &VarBinViewArray,
) -> VortexResult<()> {
    partial.merge(varbin_compute_min_max(array, array.dtype()));
    Ok(())
}

fn varbin_compute_min_max<T: ArrayAccessor<[u8]>>(
    array: &T,
    dtype: &DType,
) -> Option<MinMaxResult> {
    array.with_iterator(|iter| match iter.flatten().minmax() {
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
