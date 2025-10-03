// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_scalar::Scalar;

use crate::arrays::extension::{ExtensionArray, ExtensionVTable};
use crate::vtable::OperationsVTable;
use crate::{ArrayRef, IntoArray};

impl OperationsVTable<ExtensionVTable> for ExtensionVTable {
    fn slice(array: &ExtensionArray, range: Range<usize>) -> ArrayRef {
        ExtensionArray::new(array.ext_dtype().clone(), array.storage().slice(range)).into_array()
    }

    fn scalar_at(array: &ExtensionArray, index: usize) -> Scalar {
        Scalar::extension(array.ext_dtype().clone(), array.storage().scalar_at(index))
    }
}
