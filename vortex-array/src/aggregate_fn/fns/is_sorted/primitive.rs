// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::IsSortedIteratorExt;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;

#[derive(Copy, Clone)]
pub(super) struct ComparablePrimitive<T: NativePType>(T);

impl<T> From<&T> for ComparablePrimitive<T>
where
    T: NativePType,
{
    fn from(value: &T) -> Self {
        Self(*value)
    }
}

impl<T> PartialOrd for ComparablePrimitive<T>
where
    T: NativePType,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.total_compare(other.0))
    }
}

impl<T> PartialEq for ComparablePrimitive<T>
where
    T: NativePType,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.is_eq(other.0)
    }
}

pub(super) fn check_primitive_sorted(array: &PrimitiveArray, strict: bool) -> VortexResult<bool> {
    match_each_native_ptype!(array.ptype(), |P| { compute_is_sorted::<P>(array, strict) })
}

fn compute_is_sorted<T: NativePType>(array: &PrimitiveArray, strict: bool) -> VortexResult<bool> {
    match array.validity_mask()? {
        Mask::AllFalse(_) => Ok(!strict),
        Mask::AllTrue(_) => {
            let slice = array.as_slice::<T>();
            let iter = slice.iter().map(ComparablePrimitive::from);

            Ok(if strict {
                iter.is_strict_sorted()
            } else {
                iter.is_sorted()
            })
        }
        Mask::Values(mask_values) => {
            let iter = mask_values
                .bit_buffer()
                .iter()
                .zip_eq(array.as_slice::<T>())
                .map(|(is_valid, value)| is_valid.then_some(ComparablePrimitive::from(value)));

            Ok(if strict {
                iter.is_strict_sorted()
            } else {
                iter.is_sorted()
            })
        }
    }
}
