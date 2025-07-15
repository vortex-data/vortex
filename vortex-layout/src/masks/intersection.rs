// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::pin::Pin;

use futures::stream::{BoxStream, Stream};
use futures::task::{Context, Poll};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

/// A stream that merges multiple unaligned mask streams while performing an intersection.
pub struct IntersectionMaskStream<'a> {
    streams: Vec<BoxStream<'a, VortexResult<Mask>>>,
    // For each stream, we keep track of the current mask and the offset within it.
    next: Vec<Option<(Mask, usize)>>,
    finished: bool,
}

impl<'a> IntersectionMaskStream<'a> {
    pub fn new(streams: Vec<BoxStream<'a, VortexResult<Mask>>>) -> Self {
        assert!(!streams.is_empty(), "must have at least one stream");
        let stream_count = streams.len();
        Self {
            streams,
            next: vec![None; stream_count],
            finished: false,
        }
    }

    /// Finds the minimum remaining length across all current masks
    fn min_remaining_length(&self) -> Option<usize> {
        self.next
            .iter()
            .filter_map(|opt| opt.as_ref())
            .map(|(mask, offset)| mask.len().saturating_sub(*offset))
            .min()
    }

    /// Checks if all streams have current masks available
    fn all_masks_available(&self) -> bool {
        self.next.iter().all(|opt| opt.is_some())
    }

    /// Advances all offsets by the given amount
    fn advance_offsets(&mut self, advance_by: usize) {
        for next_item in &mut self.next {
            if let Some((mask, offset)) = next_item {
                *offset += advance_by;
                // If we've consumed the entire mask, clear it
                if *offset >= mask.len() {
                    *next_item = None;
                }
            }
        }
    }

    /// Performs intersection of current mask slices
    fn intersect_current_slices(&self, slice_length: usize) -> Mask {
        let mut result_mask = None;

        for (mask, offset) in self.next.iter().filter_map(|opt| opt.as_ref()) {
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
}

impl Stream for IntersectionMaskStream<'_> {
    type Item = VortexResult<Mask>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        loop {
            // First, try to fill any empty slots in our next array
            for i in 0..self.streams.len() {
                if self.next[i].is_none() {
                    match Pin::new(&mut self.streams[i]).poll_next(cx) {
                        Poll::Ready(Some(Ok(mask))) => {
                            self.next[i] = Some((mask, 0));
                        }
                        Poll::Ready(Some(Err(e))) => {
                            self.finished = true;
                            return Poll::Ready(Some(Err(e)));
                        }
                        Poll::Ready(None) => {
                            // Stream is exhausted
                            self.finished = true;
                            return Poll::Ready(None);
                        }
                        Poll::Pending => {
                            // Continue to check other streams
                            continue;
                        }
                    }
                }
            }

            // Check if all streams are exhausted
            if self.next.iter().all(|opt| opt.is_none()) {
                self.finished = true;
                return Poll::Ready(None);
            }

            // If we don't have masks from all streams, we can't proceed
            if !self.all_masks_available() {
                return Poll::Pending;
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

            return Poll::Ready(Some(Ok(result_mask)));
        }
    }
}

// Example usage and helper functions
#[cfg(test)]
impl<'a> IntersectionMaskStream<'a> {
    /// Convenience method to create from a vector of mask vectors (for testing)
    pub fn from_mask_vecs(mask_vecs: Vec<Vec<Mask>>) -> Self {
        use futures::{StreamExt, stream};

        let streams: Vec<BoxStream<'a, VortexResult<Mask>>> = mask_vecs
            .into_iter()
            .map(|masks| stream::iter(masks.into_iter().map(Ok)).boxed())
            .collect();

        Self::new(streams)
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use itertools::Itertools;

    use super::*;

    #[tokio::test]
    async fn test_basic_intersection() {
        // Create some test masks
        let mask1 = Mask::from_iter([true, true, false, true].iter().cloned());
        let mask2 = Mask::from_iter([true, false, true, true].iter().cloned());

        let stream = IntersectionMaskStream::from_mask_vecs(vec![vec![mask1], vec![mask2]]);

        let results: Vec<_> = stream.collect().await;
        assert_eq!(results.len(), 1);

        let result_mask = results[0].as_ref().unwrap();
        let expected = [true, false, false, true]; // Intersection
        assert_eq!(
            result_mask.to_boolean_buffer().iter().collect_vec(),
            expected
        );
    }

    #[tokio::test]
    async fn test_different_sized_masks() {
        let mask1 = Mask::from_iter([true, true].iter().cloned());
        let mask2 = Mask::from_iter([true, false].iter().cloned());
        let mask3 = Mask::from_iter([true, false, true, true].iter().cloned());

        let stream = IntersectionMaskStream::from_mask_vecs(vec![vec![mask1, mask2], vec![mask3]]);

        let results: Vec<_> = stream.collect().await;
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
