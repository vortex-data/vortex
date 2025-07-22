// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod intersection;
mod repartition;

pub use intersection::*;
pub use repartition::*;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

/// A trait for mask iterators that can be implemented by different iterator types
pub trait MaskIterator: Iterator<Item = VortexResult<Mask>> + Send {}

impl<T> MaskIterator for T where T: Iterator<Item = VortexResult<Mask>> + Send {}

/// A boxed mask iterator type that can be used as a trait object
pub type BoxMaskIterator = Box<dyn MaskIterator>;

pub trait MaskIteratorExt: MaskIterator {
    fn repartition(self, target_size: usize) -> RepartitionMaskIterator
    where
        Self: Sized + Send + 'static,
    {
        RepartitionMaskIterator::new(Box::new(self), target_size)
    }
}

impl<T> MaskIteratorExt for T where T: MaskIterator {}

pub struct AllFalseMaskIterator {
    remaining: u64,
}

impl AllFalseMaskIterator {
    fn new(value: u64) -> Self {
        AllFalseMaskIterator { remaining: value }
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

        Some(Ok(Mask::AllFalse(chunk)))
    }
}

// Helper function to create the iterator
pub fn all_false_iterator(value: u64) -> AllFalseMaskIterator {
    AllFalseMaskIterator::new(value)
}
