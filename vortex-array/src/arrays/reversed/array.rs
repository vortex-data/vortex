// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::{Array, ArrayParts, EmptyArrayData, TypedArrayRef};
use crate::arrays::Reversed;

/// Slot index for the inner (to-be-reversed) child array.
pub(super) const CHILD_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child"];

/// Extension trait for accessing [`ReversedArray`](crate::arrays::ReversedArray) properties.
pub trait ReversedArrayExt: TypedArrayRef<Reversed> {
    /// Returns the inner array whose elements will be yielded in reverse order.
    fn child(&self) -> &ArrayRef {
        self.as_ref().slots()[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated ReversedArray child slot")
    }
}

impl<T: TypedArrayRef<Reversed>> ReversedArrayExt for T {}

impl Array<Reversed> {
    /// Wraps `child` in a [`ReversedArray`](crate::arrays::ReversedArray).
    pub fn try_new(child: ArrayRef) -> VortexResult<Self> {
        let dtype = child.dtype().clone();
        let len = child.len();
        Array::try_from_parts(
            ArrayParts::new(Reversed, dtype, len, EmptyArrayData).with_slots(vec![Some(child)]),
        )
    }

    /// Wraps `child` in a [`ReversedArray`](crate::arrays::ReversedArray) without validation.
    ///
    /// # Safety
    ///
    /// Caller must ensure `child` is a valid array.
    pub unsafe fn new_unchecked(child: ArrayRef) -> Self {
        let dtype = child.dtype().clone();
        let len = child.len();
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Reversed, dtype, len, EmptyArrayData).with_slots(vec![Some(child)]),
            )
        }
    }
}
