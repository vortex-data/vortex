// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::arrays::{VarBinArray, VarBinVTable, varbin_scalar};
use crate::compute::sub_scalar;
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{Array, ArrayRef, IntoArray};

impl OperationsVTable<VarBinVTable> for VarBinVTable {
    fn slice(array: &VarBinArray, range: Range<usize>) -> ArrayRef {
        let sliced_offsets = array.offsets().slice(range.start..range.end + 1);

        // Get the first and last offset values to determine which bytes to include
        let first_offset = sliced_offsets
            .scalar_at(0)
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("Offset must be convertible to usize");
        let last_offset = sliced_offsets
            .scalar_at(sliced_offsets.len() - 1)
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("Offset must be convertible to usize");

        // Adjust offsets to start at 0 by subtracting the first offset
        let adjusted_offsets = sub_scalar(&sliced_offsets, sliced_offsets.scalar_at(0))
            .vortex_expect("Failed to adjust offsets in VarBinArray slice");

        // Slice the bytes buffer to only contain the relevant portion
        let sliced_bytes = array.bytes().slice(first_offset..last_offset);

        // SAFETY: Slicing preserves all VarBinArray invariants:
        // - Offsets remain monotonically increasing (adjusted to start at 0).
        // - Bytes buffer is sliced to match the adjusted offsets exactly.
        // - DType is preserved from the parent array.
        // - Validity is correctly sliced to match the new length.
        // - UTF-8 validity is preserved since we're slicing complete strings.
        unsafe {
            VarBinArray::new_unchecked(
                adjusted_offsets,
                sliced_bytes,
                array.dtype().clone(),
                array.validity().slice(range),
            )
        }
        .into_array()
    }

    fn scalar_at(array: &VarBinArray, index: usize) -> Scalar {
        varbin_scalar(array.bytes_at(index), array.dtype())
    }
}
