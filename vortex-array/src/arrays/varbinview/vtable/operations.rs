// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_scalar::Scalar;

use crate::arrays::{
    VarBinViewArray,
    VarBinViewVTable,
    varbin_scalar,
};
use crate::vtable::{
    OperationsVTable,
    ValidityHelper,
};
use crate::{
    ArrayRef,
    IntoArray,
};

impl OperationsVTable<VarBinViewVTable> for VarBinViewVTable {
    fn slice(array: &VarBinViewArray, range: Range<usize>) -> ArrayRef {
        let views = array.views().slice(range.clone());

        VarBinViewArray::new(
            views,
            array.buffers().clone(),
            array.dtype().clone(),
            array.validity().slice(range),
        )
        .into_array()
    }

    fn scalar_at(array: &VarBinViewArray, index: usize) -> Scalar {
        varbin_scalar(array.bytes_at(index), array.dtype())
    }
}
