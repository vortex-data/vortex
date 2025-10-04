// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::ListScalar;

use crate::arrays::{ListArray, ListVTable};
use crate::compute::{IsSortedIteratorExt, IsSortedKernel, IsSortedKernelAdapter};
use crate::register_kernel;
use crate::vtable::OperationsVTable;

impl IsSortedKernel for ListVTable {
    fn is_sorted(&self, array: &ListArray) -> VortexResult<Option<bool>> {
        // Compare lists lexicographically using the PartialOrd implementation on ListScalar.
        let scalars: Vec<Option<_>> = (0..array.len())
            .map(|i| {
                if array.is_valid(i) {
                    Some(ListVTable::scalar_at(array, i))
                } else {
                    None
                }
            })
            .collect();

        let iter = scalars
            .iter()
            .map(|opt| opt.as_ref().and_then(|s| ListScalar::try_from(s).ok()));

        Ok(Some(iter.is_sorted()))
    }

    fn is_strict_sorted(&self, array: &ListArray) -> VortexResult<Option<bool>> {
        // Compare lists lexicographically without duplicates.
        let scalars: Vec<Option<_>> = (0..array.len())
            .map(|i| {
                if array.is_valid(i) {
                    Some(ListVTable::scalar_at(array, i))
                } else {
                    None
                }
            })
            .collect();

        let iter = scalars
            .iter()
            .map(|opt| opt.as_ref().and_then(|s| ListScalar::try_from(s).ok()));

        Ok(Some(iter.is_strict_sorted()))
    }
}

register_kernel!(IsSortedKernelAdapter(ListVTable).lift());
