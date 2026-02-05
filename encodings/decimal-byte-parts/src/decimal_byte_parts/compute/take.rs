// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::arrays::TakeReduce;
use vortex_array::arrays::TakeReduceAdaptor;
use vortex_array::compute::take;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_error::VortexResult;

use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsVTable;

fn take_decimal_byte_parts(
    array: &DecimalBytePartsArray,
    indices: &dyn Array,
) -> VortexResult<ArrayRef> {
    DecimalBytePartsArray::try_new(take(&array.msp, indices)?, *array.decimal_dtype())
        .map(|a| a.to_array())
}

impl TakeReduce for DecimalBytePartsVTable {
    fn take(array: &DecimalBytePartsArray, indices: &dyn Array) -> VortexResult<Option<ArrayRef>> {
        take_decimal_byte_parts(array, indices).map(Some)
    }
}

impl DecimalBytePartsVTable {
    pub const TAKE_RULES: ParentRuleSet<Self> =
        ParentRuleSet::new(&[ParentRuleSet::lift(&TakeReduceAdaptor::<Self>(Self))]);
}
