// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::EliasFano;
use crate::array::EliasFanoArraySlotsExt;

impl CastReduce for EliasFano {
    /// Cast by rewriting the two universe bounds, and nothing else.
    ///
    /// The encoded bits never mention the ptype, so if both bounds are exactly representable in the
    /// target the span and every derived quantity are unchanged and both buffers carry over. A cast
    /// that cannot represent a bound returns `None` and leaves it to the generic path.
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !dtype.is_int() || dtype.is_nullable() {
            return Ok(None);
        }
        if dtype == array.array().dtype() {
            return Ok(Some(array.array().clone()));
        }

        let data = array.data();
        let (Ok(reference), Ok(max)) = (
            data.reference_scalar().cast(dtype),
            data.max_scalar().cast(dtype),
        ) else {
            return Ok(None);
        };

        Ok(Some(
            EliasFano::try_new(
                data.clone().with_bounds(reference, max),
                array.lower().clone(),
                array.len(),
            )?
            .into_array(),
        ))
    }
}
