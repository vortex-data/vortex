// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::compute::MaskReduce;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsVTable;

impl MaskReduce for DecimalBytePartsVTable {
    fn mask(array: &DecimalBytePartsArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_msp =
            MaskedArray::try_new(array.msp.clone(), Validity::Array(mask.clone()))?.into_array();
        Ok(Some(
            DecimalBytePartsArray::try_new(masked_msp, *array.decimal_dtype())?.into_array(),
        ))
    }
}
