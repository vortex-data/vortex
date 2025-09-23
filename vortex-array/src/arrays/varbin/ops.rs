// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_scalar::Scalar;

use crate::arrays::{VarBinArray, VarBinVTable, varbin_scalar};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{Array, ArrayRef, IntoArray};

impl OperationsVTable<VarBinVTable> for VarBinVTable {
    fn slice(array: &VarBinArray, range: Range<usize>) -> ArrayRef {
        unsafe {
            VarBinArray::new_unchecked(
                array.offsets().slice(range.start..range.end + 1),
                array.bytes().clone(),
                array.dtype().clone(),
                array.validity().slice(range),
            )
            .into_array()
        }
    }

    fn scalar_at(array: &VarBinArray, index: usize) -> Scalar {
        varbin_scalar(array.bytes_at(index), array.dtype())
    }
}
