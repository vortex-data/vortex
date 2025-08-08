// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;

use crate::arrays::{VarBinArray, VarBinVTable, varbin_scalar};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{Array, ArrayRef, IntoArray};

impl OperationsVTable<VarBinVTable> for VarBinVTable {
    fn slice(array: &VarBinArray, start: usize, stop: usize) -> ArrayRef {
        VarBinArray::new(
            array.offsets().slice(start, stop + 1),
            array.bytes().clone(),
            array.dtype().clone(),
            array.validity().slice(start, stop),
        )
        .into_array()
    }

    fn scalar_at(array: &VarBinArray, index: usize) -> Scalar {
        varbin_scalar(array.bytes_at(index), array.dtype())
    }
}
