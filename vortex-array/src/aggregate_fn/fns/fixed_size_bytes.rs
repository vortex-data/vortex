// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-size lists of non-nullable `u8`, which order lexicographically by byte.
//!
//! This is the storage of byte-string extension types such as UUID, so order-based aggregates can
//! support it without implementing general list comparison.

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;

/// Whether `dtype` is a fixed-size list of non-nullable `u8`.
pub(crate) fn is_fixed_size_bytes(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::FixedSizeList(element_dtype, ..)
            if matches!(element_dtype.as_ref(), DType::Primitive(PType::U8, Nullability::NonNullable))
    )
}

/// The values of a fixed-size list of bytes, each one a slice of the decoded elements.
pub(crate) struct FixedSizeBytes {
    elements: PrimitiveArray,
    size: usize,
    len: usize,
}

impl FixedSizeBytes {
    /// Decode the elements of `array`, which must be a fixed-size list of bytes.
    pub(crate) fn try_new(
        array: &FixedSizeListArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        Ok(Self {
            elements: array.elements().clone().execute::<PrimitiveArray>(ctx)?,
            size: array.list_size() as usize,
            len: array.len(),
        })
    }

    /// Every value in order, including the values of null lists.
    pub(crate) fn values(&self) -> impl Iterator<Item = &[u8]> {
        let bytes = self.elements.as_slice::<u8>();
        (0..self.len).map(move |index| &bytes[index * self.size..(index + 1) * self.size])
    }
}
