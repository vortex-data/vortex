// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::arrays::varbin_scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<VarBinViewVTable> for VarBinViewVTable {
    fn slice(_array: &VarBinViewArray, _range: Range<usize>) -> ArrayRef {
        unreachable!("replaced with SliceArray")
    }

    fn scalar_at(array: &VarBinViewArray, index: usize) -> Scalar {
        varbin_scalar(array.bytes_at(index), array.dtype())
    }
}
