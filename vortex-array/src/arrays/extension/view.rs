// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::arrays::ExtensionArray;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtVTable;

/// A typed view of an extension array.
pub struct ExtArray<'a, V: ExtVTable> {
    ext_dtype: &'a ExtDType<V>,
    array: &'a ExtensionArray,
}

impl<'a, V: ExtVTable> ExtArray<'a, V> {
    pub fn try_new(array: &'a ExtensionArray) -> Option<Self> {
        let ext_dtype = array.ext_dtype().downcast_ref::<V>()?;
        Some(Self { ext_dtype, array })
    }

    pub fn ext_dtype(&self) -> &ExtDType<V> {
        self.ext_dtype
    }

    pub fn storage_array(&self) -> &ArrayRef {
        self.array.storage_array()
    }
}
