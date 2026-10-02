// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Narrow;
use super::NarrowArray;
use super::NarrowArraySlotsExt;
use crate::ArrayRef;
use crate::ArrayView;
use crate::IntoArray;
use crate::arrays::dict::TakeReduce;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduce;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullReduce;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(super) const RULES: ParentRuleSet<Narrow> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&FillNullReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&FilterReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Narrow)),
]);

fn rewrap(array: ArrayView<'_, Narrow>, values: ArrayRef) -> VortexResult<Option<ArrayRef>> {
    let dtype = array.dtype().with_nullability(values.dtype().nullability());
    Ok(Some(NarrowArray::try_new(values, dtype)?.into_array()))
}

impl SliceReduce for Narrow {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().slice(range)?)
    }
}

impl FilterReduce for Narrow {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().filter(mask.clone())?)
    }
}

impl TakeReduce for Narrow {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().take(indices.clone())?)
    }
}

impl MaskReduce for Narrow {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().clone().mask(mask.clone())?)
    }
}

impl CastReduce for Narrow {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !dtype.is_int() || dtype.is_signed_int() != array.dtype().is_signed_int() {
            return Ok(None);
        }

        let storage_dtype = array.values().dtype().with_nullability(dtype.nullability());
        if dtype.as_ptype().byte_width() <= storage_dtype.as_ptype().byte_width() {
            return array.values().cast(dtype.clone()).map(Some);
        }

        Ok(Some(
            NarrowArray::try_new(array.values().cast(storage_dtype)?, dtype.clone())?.into_array(),
        ))
    }
}

impl FillNullReduce for Narrow {
    fn fill_null(array: ArrayView<'_, Self>, fill_value: &Scalar) -> VortexResult<Option<ArrayRef>> {
        let storage_dtype = array
            .values()
            .dtype()
            .with_nullability(fill_value.dtype().nullability());
        let Ok(fill_value) = fill_value.cast(&storage_dtype) else {
            // A valid logical value may require wider storage. Let canonical execution handle it.
            return Ok(None);
        };

        rewrap(array, array.values().fill_null(fill_value)?)
    }
}
