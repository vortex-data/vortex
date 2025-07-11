// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;

use futures::stream::{BoxStream, Stream};
use futures::task::{Context, Poll};
use vortex_error::VortexResult;
use vortex_mask::Mask;

pub struct RepartitionMaskStream<'a> {
    source: BoxStream<'a, VortexResult<Mask>>,
    target_row_count: usize,

    // Buffer to accumulate masks until we have enough for a complete chunk
    buffer: Vec<Mask>,
    buffer_row_count: usize,

    // Current mask being processed and offset within it
    current_mask: Option<Mask>,
    current_offset: usize,

    finished: bool,
}

impl<'a> RepartitionMaskStream<'a> {
    pub fn new(source: BoxStream<'a, VortexResult<Mask>>, target_row_count: usize) -> Self {
        assert!(target_row_count > 0, "Target row count must be positive");

        Self {
            source,
            target_row_count,
            buffer: Vec::new(),
            buffer_row_count: 0,
            current_mask: None,
            current_offset: 0,
            finished: false,
        }
    }

    /// Adds a mask to the buffer, potentially splitting it if it would exceed target_row_count
    fn add_to_buffer(&mut self, mask: Mask) -> Option<Mask> {
        let mask_len = mask.len();
        let remaining_capacity = self.target_row_count - self.buffer_row_count;

        if mask_len <= remaining_capacity {
            // Entire mask fits in current buffer
            self.buffer.push(mask);
            self.buffer_row_count += mask_len;

            if self.buffer_row_count == self.target_row_count {
                // Buffer is complete, return the concatenated mask
                Some(self.flush_buffer())
            } else {
                // Buffer not yet complete
                None
            }
        } else {
            // Mask needs to be split
            let first_part = mask.slice(0, remaining_capacity);
            let second_part = mask.slice(remaining_capacity, mask_len - remaining_capacity);

            // Add first part to complete the buffer
            self.buffer.push(first_part);
            self.buffer_row_count += remaining_capacity;

            // Store the second part for next iteration
            self.current_mask = Some(second_part);
            self.current_offset = 0;

            // Return the completed buffer
            Some(self.flush_buffer())
        }
    }

    /// Flushes the current buffer and returns the concatenated mask
    fn flush_buffer(&mut self) -> Mask {
        assert!(
            !self.buffer.is_empty(),
            "Buffer should not be empty when flushing"
        );

        if self.buffer.len() == 1 {
            // Optimization: if only one mask, return it directly
            let mask = self.buffer.pop().unwrap();
            self.buffer_row_count = 0;
            mask
        } else {
            // Concatenate multiple masks
            let masks = std::mem::take(&mut self.buffer);
            self.buffer_row_count = 0;
            Mask::from_iter(masks.into_iter())
        }
    }

    /// Processes the current mask in the buffer, potentially yielding a complete chunk
    fn process_current_mask(&mut self) -> Option<Mask> {
        if let Some(mask) = self.current_mask.take() {
            let mask_len = mask.len();
            let remaining_capacity = self.target_row_count - self.buffer_row_count;

            if mask_len <= remaining_capacity {
                // Entire remaining mask fits in current buffer
                self.buffer.push(mask);
                self.buffer_row_count += mask_len;
                self.current_offset = 0;

                if self.buffer_row_count == self.target_row_count {
                    // Buffer is complete
                    Some(self.flush_buffer())
                } else {
                    // Buffer not yet complete
                    None
                }
            } else {
                // Need to split the mask
                let first_part = mask.slice(0, remaining_capacity);
                let second_part = mask.slice(remaining_capacity, mask_len - remaining_capacity);

                // Add first part to complete the buffer
                self.buffer.push(first_part);
                self.buffer_row_count += remaining_capacity;

                // Keep the second part for next iteration
                self.current_mask = Some(second_part);
                self.current_offset = 0;

                // Return the completed buffer
                Some(self.flush_buffer())
            }
        } else {
            None
        }
    }

    /// Finishes the stream by returning any remaining buffered data
    fn finish(&mut self) -> Option<VortexResult<Mask>> {
        if self.buffer_row_count > 0 {
            Some(Ok(self.flush_buffer()))
        } else {
            None
        }
    }
}

impl<'a> Stream for RepartitionMaskStream<'a> {
    type Item = VortexResult<Mask>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        loop {
            // First, try to process any current mask we have buffered
            if self.current_mask.is_some() {
                match self.process_current_mask() {
                    Some(mask) => return Poll::Ready(Some(Ok(mask))),
                    None => {
                        // Continue to fetch more data
                    }
                }
            }

            // Try to get the next mask from the source stream
            match Pin::new(&mut self.source).poll_next(cx) {
                Poll::Ready(Some(Ok(mask))) => {
                    match self.add_to_buffer(mask) {
                        Some(result_mask) => {
                            return Poll::Ready(Some(Ok(result_mask)));
                        }
                        None => {
                            // Buffer not yet complete, continue polling
                            continue;
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    // Source stream is exhausted
                    self.finished = true;
                    return Poll::Ready(self.finish());
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt, stream};

    use super::*;

    #[tokio::test]
    async fn test_exact_repartition() {
        // Input: masks of size [2, 2, 2], target: 3
        // Expected: [3, 3] (first 3 elements, then next 3)
        let masks = vec![
            Mask::from_iter([true, false].iter().cloned()),
            Mask::from_iter([true, true].iter().cloned()),
            Mask::from_iter([false, true].iter().cloned()),
        ];

        let source = stream::iter(masks.into_iter().map(Ok)).boxed();
        let repartition_stream = RepartitionMaskStream::new(source, 3);

        let results: Vec<_> = repartition_stream.collect().await;
        assert_eq!(results.len(), 2);

        let mask1 = results[0].as_ref().unwrap();
        let mask2 = results[1].as_ref().unwrap();

        assert_eq!(mask1.len(), 3);
        assert_eq!(mask2.len(), 3);

        // First mask should be [true, false, true]
        assert_eq!(mask1.to_vec(), [true, false, true]);
        // Second mask should be [true, false, true]
        assert_eq!(mask2.to_vec(), [true, false, true]);
    }

    #[tokio::test]
    async fn test_uneven_repartition() {
        // Input: masks of size [3, 2], target: 2
        // Expected: [2, 2, 1] (first 2 elements, next 2, last 1)
        let masks = vec![
            Mask::from_iter([true, false, true].iter().cloned()),
            Mask::from_iter([false, true].iter().cloned()),
        ];

        let source = stream::iter(masks.into_iter().map(Ok)).boxed();
        let repartition_stream = RepartitionMaskStream::new(source, 2);

        let results: Vec<_> = repartition_stream.collect().await;
        assert_eq!(results.len(), 3);

        let mask1 = results[0].as_ref().unwrap();
        let mask2 = results[1].as_ref().unwrap();
        let mask3 = results[2].as_ref().unwrap();

        assert_eq!(mask1.len(), 2);
        assert_eq!(mask2.len(), 2);
        assert_eq!(mask3.len(), 1);

        assert_eq!(mask1.to_vec(), [true, false]);
        assert_eq!(mask2.to_vec(), [true, false]);
        assert_eq!(mask3.to_vec(), [true]);
    }

    #[tokio::test]
    async fn test_larger_target_than_input() {
        // Input: masks of size [2, 1], target: 5
        // Expected: [3] (all elements combined, less than target)
        let masks = vec![
            Mask::from_iter([true, false].iter().cloned()),
            Mask::from_iter([true].iter().cloned()),
        ];

        let source = stream::iter(masks.into_iter().map(Ok)).boxed();
        let repartition_stream = RepartitionMaskStream::new(source, 5);

        let results: Vec<_> = repartition_stream.collect().await;
        assert_eq!(results.len(), 1);

        let mask = results[0].as_ref().unwrap();
        assert_eq!(mask.len(), 3);
        assert_eq!(mask.to_vec(), [true, false, true]);
    }

    #[tokio::test]
    async fn test_single_large_mask() {
        // Input: single mask of size 7, target: 3
        // Expected: [3, 3, 1]
        let mask = Mask::from_iter(
            [true, false, true, false, true, false, true]
                .iter()
                .cloned(),
        );

        let source = stream::iter(vec![mask].into_iter().map(Ok)).boxed();
        let repartition_stream = RepartitionMaskStream::new(source, 3);

        let results: Vec<_> = repartition_stream.collect().await;
        assert_eq!(results.len(), 3);

        let mask1 = results[0].as_ref().unwrap();
        let mask2 = results[1].as_ref().unwrap();
        let mask3 = results[2].as_ref().unwrap();

        assert_eq!(mask1.len(), 3);
        assert_eq!(mask2.len(), 3);
        assert_eq!(mask3.len(), 1);

        assert_eq!(mask1.to_vec(), [true, false, true]);
        assert_eq!(mask2.to_vec(), [false, true, false]);
        assert_eq!(mask3.to_vec(), [true]);
    }

    #[tokio::test]
    async fn test_empty_stream() {
        let source = stream::iter(Vec::<VortexResult<Mask>>::new()).boxed();
        let repartition_stream = RepartitionMaskStream::new(source, 3);

        let results: Vec<_> = repartition_stream.collect().await;
        assert_eq!(results.len(), 0);
    }
}
