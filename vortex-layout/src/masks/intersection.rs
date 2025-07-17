// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use super::BoxMaskIterator;

/// An iterator that merges multiple mask iterators while performing an intersection.
pub struct IntersectionMaskIterator {
    iterators: Vec<BoxMaskIterator>,
    // For each iterator, we keep track of the current mask and the offset within it.
    current: Vec<Option<(Mask, usize)>>,
    finished: bool,
}

impl IntersectionMaskIterator {
    pub fn new(iterators: Vec<BoxMaskIterator>) -> Self {
        assert!(!iterators.is_empty(), "must have at least one iterator");
        let iter_count = iterators.len();
        Self {
            iterators,
            current: vec![None; iter_count],
            finished: false,
        }
    }

    /// Finds the minimum remaining length across all current masks
    fn min_remaining_length(&self) -> Option<usize> {
        self.current
            .iter()
            .filter_map(|opt| opt.as_ref())
            .map(|(mask, offset)| mask.len().saturating_sub(*offset))
            .min()
    }

    /// Checks if all iterators have current masks available
    fn all_masks_available(&self) -> bool {
        self.current.iter().all(|opt| opt.is_some())
    }

    /// Advances all offsets by the given amount
    fn advance_offsets(&mut self, advance_by: usize) {
        for current_item in &mut self.current {
            if let Some((mask, offset)) = current_item {
                *offset += advance_by;
                // If we've consumed the entire mask, clear it
                if *offset >= mask.len() {
                    *current_item = None;
                }
            }
        }
    }

    /// Performs intersection of current mask slices
    fn intersect_current_slices(&self, slice_length: usize) -> Mask {
        let mut result_mask = None;

        for (mask, offset) in self.current.iter().filter_map(|opt| opt.as_ref()) {
            let slice = mask.slice(*offset, slice_length);

            match result_mask {
                None => result_mask = Some(slice),
                Some(current) => {
                    result_mask = Some(current.bitand(&slice));
                }
            }
        }

        result_mask.vortex_expect("no masks")
    }

    /// Tries to fill any empty slots in the current array
    /// Returns true if any iterator is exhausted (which means intersection is complete)
    fn try_fill_empty_slots(&mut self) -> VortexResult<bool> {
        let mut any_exhausted = false;
        for i in 0..self.iterators.len() {
            if self.current[i].is_none() {
                match self.iterators[i].next() {
                    Some(Ok(mask)) => {
                        self.current[i] = Some((mask, 0));
                    }
                    Some(Err(e)) => {
                        return Err(e);
                    }
                    None => {
                        // Iterator is exhausted - for intersection, if ANY iterator is exhausted, 
                        // the intersection is complete
                        any_exhausted = true;
                    }
                }
            }
        }
        Ok(any_exhausted)
    }
}

impl Iterator for IntersectionMaskIterator {
    type Item = VortexResult<Mask>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        loop {
            // Try to fill any empty slots
            let any_exhausted = match self.try_fill_empty_slots() {
                Ok(exhausted) => exhausted,
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            };

            // If any iterator is exhausted, intersection is complete
            if any_exhausted {
                self.finished = true;
                return None;
            }

            // Check if all iterators are exhausted
            if self.current.iter().all(|opt| opt.is_none()) {
                self.finished = true;
                return None;
            }

            // If we don't have masks from all iterators, we can't proceed
            if !self.all_masks_available() {
                // This should not happen if try_fill_empty_slots worked correctly
                self.finished = true;
                return None;
            }

            // Find the minimum remaining length
            let min_length = match self.min_remaining_length() {
                Some(len) if len > 0 => len,
                _ => {
                    // All current masks are exhausted, continue to fetch new ones
                    continue;
                }
            };

            // Perform intersection on the slice
            let result_mask = self.intersect_current_slices(min_length);
            // Advance all offsets
            self.advance_offsets(min_length);

            return Some(Ok(result_mask));
        }
    }
}

#[cfg(test)]
impl IntersectionMaskIterator {
    /// Convenience method to create from a vector of mask vectors (for testing)
    pub fn from_mask_vecs(mask_vecs: Vec<Vec<Mask>>) -> Self {
        let iterators: Vec<BoxMaskIterator> = mask_vecs
            .into_iter()
            .map(|masks| Box::new(masks.into_iter().map(Ok)) as BoxMaskIterator)
            .collect();

        Self::new(iterators)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;
    #[test]
    fn test_iterator_basic_intersection() {
        // Create some test masks
        let mask1 = Mask::from_iter([true, true, false, true].iter().cloned());
        let mask2 = Mask::from_iter([true, false, true, true].iter().cloned());

        let iterator = IntersectionMaskIterator::from_mask_vecs(vec![vec![mask1], vec![mask2]]);

        let results: Vec<_> = iterator.collect();
        assert_eq!(results.len(), 1);

        let result_mask = results[0].as_ref().unwrap();
        let expected = [true, false, false, true]; // Intersection
        assert_eq!(
            result_mask.to_boolean_buffer().iter().collect_vec(),
            expected
        );
    }

    #[test]
    fn test_iterator_different_sized_masks() {
        let mask1 = Mask::from_iter([true, true].iter().cloned());
        let mask2 = Mask::from_iter([true, false].iter().cloned());
        let mask3 = Mask::from_iter([true, false, true, true].iter().cloned());

        let iterator =
            IntersectionMaskIterator::from_mask_vecs(vec![vec![mask1, mask2], vec![mask3]]);

        let results: Vec<_> = iterator.collect();
        assert_eq!(results.len(), 2);

        let result = results
            .into_iter()
            .flat_map(|mask| {
                mask.unwrap()
                    .to_boolean_buffer()
                    .iter()
                    .collect_vec()
                    .into_iter()
            })
            .collect_vec();
        let expected = [true, false, true, false];
        assert_eq!(result, expected);
    }
}
