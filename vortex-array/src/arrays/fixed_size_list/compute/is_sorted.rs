// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use vortex_error::VortexResult;
use vortex_scalar::ListScalar;

use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::compute::IsSortedKernel;
use crate::compute::IsSortedKernelAdapter;
use crate::register_kernel;

/// Implementation of IsSortedKernel for FixedSizeListArray.
///
/// This implementation uses lexicographic comparison of list elements.
/// Since all lists have the same fixed size, comparison is straightforward element-wise.
/// Null lists are considered the smallest values.
/// Non-comparable lists (which shouldn't occur for lists with the same element type)
/// are treated as making the array not sorted.
impl IsSortedKernel for FixedSizeListVTable {
    fn is_sorted(&self, array: &FixedSizeListArray) -> VortexResult<Option<bool>> {
        if array.len() <= 1 {
            return Ok(Some(true));
        }
        for i in 0..array.len() - 1 {
            let scalar_a = array.scalar_at(i)?;
            let scalar_b = array.scalar_at(i + 1)?;
            let a = ListScalar::try_from(&scalar_a)?;
            let b = ListScalar::try_from(&scalar_b)?;
            // For is_sorted, we allow Less and Equal, but not Greater or incomparable (None)
            match a.partial_cmp(&b) {
                Some(Ordering::Greater) | None => return Ok(Some(false)),
                _ => {}
            }
        }
        Ok(Some(true))
    }

    fn is_strict_sorted(&self, array: &FixedSizeListArray) -> VortexResult<Option<bool>> {
        if array.len() <= 1 {
            return Ok(Some(true));
        }
        for i in 0..array.len() - 1 {
            let scalar_a = array.scalar_at(i)?;
            let scalar_b = array.scalar_at(i + 1)?;
            let a = ListScalar::try_from(&scalar_a)?;
            let b = ListScalar::try_from(&scalar_b)?;
            // For is_strict_sorted, we only allow Less, not Equal, Greater, or incomparable (None)
            match a.partial_cmp(&b) {
                Some(Ordering::Greater | Ordering::Equal) | None => return Ok(Some(false)),
                _ => {}
            }
        }
        Ok(Some(true))
    }
}

register_kernel!(IsSortedKernelAdapter(FixedSizeListVTable).lift());
