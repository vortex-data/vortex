// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::Slice;

/// The underlying child array being sliced.
pub(super) const CHILD_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child"];

#[derive(Clone, Debug)]
pub struct SliceData {
    pub(super) range: Range<usize>,
}

impl Display for SliceData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "range: {}..{}", self.range.start, self.range.end)
    }
}

pub struct SliceDataParts {
    pub range: Range<usize>,
}

pub trait SliceArrayExt: TypedArrayRef<Slice> {
    fn child(&self) -> &ArrayRef {
        self.as_ref().slots()[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated slice child slot")
    }
}
impl<T: TypedArrayRef<Slice>> SliceArrayExt for T {}

impl SliceData {
    fn try_new(child_len: usize, range: Range<usize>) -> VortexResult<Self> {
        if range.end > child_len {
            vortex_panic!(
                "SliceArray range out of bounds: range {:?} exceeds child array length {}",
                range,
                child_len
            );
        }
        Ok(Self { range })
    }

    pub fn new(range: Range<usize>) -> Self {
        Self { range }
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The range used to slice the child array.
    pub fn slice_range(&self) -> &Range<usize> {
        &self.range
    }

    pub fn into_parts(self) -> SliceDataParts {
        SliceDataParts { range: self.range }
    }
}

impl Array<Slice> {
    /// Constructs a new `SliceArray`.
    pub fn try_new(child: ArrayRef, range: Range<usize>) -> VortexResult<Self> {
        let len = range.len();
        let dtype = child.dtype().clone();
        let data = SliceData::try_new(child.len(), range)?;
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Slice, dtype, len, data).with_slots(vec![Some(child)]),
            )
        })
    }

    /// Constructs a new `SliceArray`.
    pub fn new(child: ArrayRef, range: Range<usize>) -> Self {
        let len = range.len();
        let dtype = child.dtype().clone();
        let data = SliceData::new(range);
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Slice, dtype, len, data).with_slots(vec![Some(child)]),
            )
        }
    }
}
