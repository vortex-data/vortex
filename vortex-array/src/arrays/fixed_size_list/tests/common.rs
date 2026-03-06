// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Common test utilities for FixedSizeList tests.

use vortex_buffer::buffer;

use crate::IntoArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::Nullability;
use crate::validity::Validity;

/// Creates a basic FSL for testing: [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
pub fn create_basic_fsl() -> FixedSizeListArray {
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array();
    FixedSizeListArray::new(elements, 3, Validity::NonNullable, 4)
}

/// Creates a nullable FSL: [[1,2,3], null, [7,8,9]]
pub fn create_nullable_fsl() -> FixedSizeListArray {
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    let validity = Validity::from_iter([true, false, true]);
    FixedSizeListArray::new(elements, 3, validity, 3)
}

/// Creates a large FSL for performance testing (30 lists of 10 elements each)
pub fn create_large_fsl() -> FixedSizeListArray {
    let elements: Vec<i64> = (0..300).collect();
    let elements = PrimitiveArray::from_iter(elements).into_array();
    FixedSizeListArray::new(elements, 10, Validity::NonNullable, 30)
}

/// Creates a single-element FSL: [[42,43,44,45,46]]
pub fn create_single_element_fsl() -> FixedSizeListArray {
    let elements = buffer![42u32, 43, 44, 45, 46].into_array();
    FixedSizeListArray::new(elements, 5, Validity::NonNullable, 1)
}

/// Creates an empty FSL (0 lists)
pub fn create_empty_fsl() -> FixedSizeListArray {
    let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    FixedSizeListArray::new(elements.into_array(), 3, Validity::NonNullable, 0)
}
