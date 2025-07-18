// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod intersection;
mod repartition;

pub use intersection::*;
pub use repartition::*;
use vortex_error::VortexResult;
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
