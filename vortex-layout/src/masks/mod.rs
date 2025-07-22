// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod intersection;
mod repartition;

pub use intersection::*;
pub use repartition::*;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

/// A trait for mask iterators that can be implemented by different iterator types
pub trait MaskIterator: Iterator<Item = VortexResult<Mask>> {}

impl<T> MaskIterator for T where T: Iterator<Item = VortexResult<Mask>> {}

/// A boxed mask iterator type that can be used as a trait object
pub type BoxMaskIterator<'a> = Box<dyn MaskIterator + 'a>;

pub trait MaskIteratorExt: MaskIterator {
    fn repartition<'a>(self, target_size: usize) -> RepartitionMaskIterator<'a>
    where
        Self: Sized + 'a,
    {
        RepartitionMaskIterator::new(Box::new(self), target_size)
    }
}

impl<T> MaskIteratorExt for T where T: MaskIterator {}

pub struct AllFalseMaskIterator {
    remaining: u64,
    value: bool,
}

impl AllFalseMaskIterator {
    fn new(count: u64, value: bool) -> Self {
        AllFalseMaskIterator {
            remaining: count,
            value,
        }
    }
}

impl Iterator for AllFalseMaskIterator {
    type Item = VortexResult<Mask>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let chunk = if self.remaining > usize::MAX as u64 {
            self.remaining -= usize::MAX as u64;
            usize::MAX
        } else {
            let final_chunk =
                usize::try_from(self.remaining).vortex_expect("index does not fit into a usize");
            self.remaining = 0;
            final_chunk
        };

        if self.value {
            Some(Ok(Mask::AllTrue(chunk)))
        } else {
            Some(Ok(Mask::AllFalse(chunk)))
        }
    }
}

// Helper function to create the iterator
pub fn all_constant_mask_iterator(count: u64, value: bool) -> AllFalseMaskIterator {
    AllFalseMaskIterator::new(count, value)
}
