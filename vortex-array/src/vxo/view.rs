// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vxo::Array2;
use crate::vxo::VTable;
use std::ops::Deref;
use vortex_error::VortexExpect;

/// A typed view over a Vortex array.
pub struct ArrayView<'a, V: VTable> {
    array: &'a Array2,
    vtable: &'a V,
    data: &'a V::Instance,
}

impl<'a, V: VTable> ArrayView<'a, V> {
    /// Creates a new array view.
    ///
    /// # Panics
    ///
    /// If the array cannot be downcast to the specified
    pub fn new(array: &'a Array2) -> Self {
        Self::maybe_new(array).vortex_expect("Failed to downcast array to specified vtable type")
    }

    /// Creates a new array view, returns `None` if the array cannot be downcast to the specified
    /// vtable type.
    pub fn maybe_new(array: &'a Array2) -> Option<Self> {
        let vtable = array.vtable().as_dyn().as_any().downcast_ref::<V>()?;
        let data = array.data().downcast_ref::<V::Instance>()?;
        Some(Self {
            array,
            vtable,
            data,
        })
    }

    /// Returns the underlying array.
    pub fn array(&self) -> &'a Array2 {
        self.array
    }

    /// Returns the vtable for this array.
    pub fn vtable(&self) -> &'a V {
        self.vtable
    }

    /// Returns the instance data for this array.
    pub fn data(&self) -> &'a V::Instance {
        self.data
    }
}

impl<'a, V: VTable> Deref for ArrayView<'a, V> {
    type Target = Array2;

    fn deref(&self) -> &Self::Target {
        self.array
    }
}
