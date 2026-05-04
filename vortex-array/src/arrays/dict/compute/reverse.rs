// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray as _;
use crate::array::ArrayView;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::dict::DictArraySlotsExt as _;
use crate::arrays::reversed::ReverseReduce;

/// Reverses a `DictArray` by reversing only the codes array.
///
/// The values dictionary is reused unchanged.  Since codes are typically small
/// integers (`u8` or `u16`), the reversal is O(n_codes) rather than O(n_rows × value_size).
///
/// # Example
///
/// For `Dict(codes=[2,2,1,1,0,0], values=[A, B, C])` → decoded `[C,C,B,B,A,A]`:
/// `Dict(codes=[0,0,1,1,2,2], values=[A, B, C])` → decoded `[A,A,B,B,C,C]` ✓
impl ReverseReduce for Dict {
    fn reverse(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        let reversed_codes = array.codes().reverse()?;
        // SAFETY: reversing codes doesn't change the dict invariants; the values
        // dictionary is untouched and all code indices remain valid.
        Ok(Some(
            unsafe { DictArray::new_unchecked(reversed_codes, array.values().clone()) }
                .into_array(),
        ))
    }
}
