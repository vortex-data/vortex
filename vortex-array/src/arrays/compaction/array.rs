// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::smallvec;
use vortex_error::VortexExpect;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array::vtable::EmptyArrayData;
use crate::arrays::Compaction;

/// The single child array to be compacted.
pub(super) const CHILD_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child"];

/// Extension trait for accessing the child of a [`CompactionArray`](super::CompactionArray).
pub trait CompactionArrayExt: TypedArrayRef<Compaction> {
    /// The child array that will be compacted when this array is executed.
    fn child(&self) -> &ArrayRef {
        self.as_ref().slots()[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated compaction child slot")
    }
}
impl<T: TypedArrayRef<Compaction>> CompactionArrayExt for T {}

impl Array<Compaction> {
    /// Wrap `child` in a [`Compaction`] array.
    ///
    /// Executing the resulting array produces a fully-normalized version of `child` (see the
    /// [module docs](super) for the exact normalization rules). The logical type and length are
    /// preserved.
    pub fn new(child: ArrayRef) -> Self {
        let dtype = child.dtype().clone();
        let len = child.len();
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Compaction, dtype, len, EmptyArrayData)
                    .with_slots(smallvec![Some(child)]),
            )
        }
    }
}
