#![allow(dead_code)]
use std::future::Future;
use std::io;
use std::io::ErrorKind;
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use futures_util::{stream, FutureExt, StreamExt, TryStreamExt};
use vortex_error::VortexExpect;

use crate::{Dispatch, IoDispatcher, VortexReadAt};

const MAX_BUFFERED_READS: usize = 10;

#[derive(Debug, Clone)]
pub struct VortexReadRanges<R> {
    read: R,
    dispatcher: Arc<IoDispatcher>,
    max_gap: usize,
}

impl<R> VortexReadRanges<R> {
    pub fn new(read: R, dispatcher: Arc<IoDispatcher>, max_gap: usize) -> VortexReadRanges<R> {
        Self {
            read,
            dispatcher,
            max_gap,
        }
    }
}

impl<R: VortexReadAt> VortexReadRanges<R> {
    pub fn read_byte_ranges(
        &self,
        ranges: Vec<Range<usize>>,
    ) -> impl Future<Output = io::Result<Vec<Bytes>>> + Send + 'static {
        let dispatcher = self.dispatcher.clone();
        let reader = self.read.clone();
        let max_gap = self.max_gap;
        async move {
            let merged_ranges = merge_ranges(ranges.clone(), max_gap);
            let read_ranges = stream::iter(merged_ranges.iter().cloned())
                .map(|r| {
                    dispatcher
                        .dispatch({
                            let reader = reader.clone();
                            move || async move {
                                reader
                                    .read_byte_range(r.start as u64, (r.end - r.start) as u64)
                                    .await
                            }
                        })
                        .vortex_expect("dispatch async")
                        .map(|bytes| {
                            bytes
                                .map_err(|e| io::Error::new(ErrorKind::Other, e))
                                .and_then(|b| b)
                        })
                })
                .buffered(MAX_BUFFERED_READS)
                .try_collect::<Vec<_>>()
                .await?;

            let mut result_bytes = Vec::with_capacity(ranges.len());
            for range in ranges {
                let read_idx = merged_ranges.partition_point(|mr| mr.start <= range.start) - 1;

                let read_range_start = merged_ranges[read_idx].start;
                let read_bytes = &read_ranges[read_idx];
                let start = range.start - read_range_start;
                let end = range.end - read_range_start;
                result_bytes.push(read_bytes.slice(start..end.min(read_bytes.len())));
            }

            Ok(result_bytes)
        }
    }
}

fn merge_ranges(mut ranges: Vec<Range<usize>>, max_gap: usize) -> Vec<Range<usize>> {
    if ranges.is_empty() {
        return Vec::new();
    }

    ranges.sort_unstable_by_key(|r| r.start);
    let mut merged_ranges = Vec::with_capacity(ranges.len());

    let mut start_idx = 0;
    let mut end_idx = 1;

    while start_idx < ranges.len() {
        let mut range_end = ranges[start_idx].end;

        while end_idx < ranges.len()
            && ranges[end_idx]
                .start
                .checked_sub(range_end)
                .map(|gap| gap <= max_gap)
                .unwrap_or(true)
        {
            range_end = range_end.max(ranges[end_idx].end);
            end_idx += 1;
        }

        merged_ranges.push(ranges[start_idx].start..range_end);
        start_idx = end_idx;
        end_idx += 1;
    }

    merged_ranges
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;

    use crate::read_ranges::merge_ranges;
    use crate::{IoDispatcher, VortexReadRanges};

    #[test]
    fn merges_ranges() {
        let ranges = vec![0..2, 12..20];
        let merged = merge_ranges(ranges, 10);
        assert_eq!(merged, vec![0..20]);
    }

    #[test]
    fn avoids_merging() {
        let ranges = vec![0..2, 12..20];
        let merged = merge_ranges(ranges, 5);
        assert_eq!(merged, vec![0..2, 12..20]);
    }

    #[tokio::test]
    async fn read_ranges() {
        let bytes = Bytes::from("trytoreadthisinmultiplechunks");
        let range_read = VortexReadRanges::new(bytes, Arc::new(IoDispatcher::new_tokio(1)), 15);
        let ranges = vec![5..9, 23..29];
        let merged_ranges = merge_ranges(ranges.clone(), 15);
        assert_eq!(merged_ranges, vec![5..29]);
        let read_ranges = range_read.read_byte_ranges(ranges).await.unwrap();
        assert_eq!(
            read_ranges,
            vec![Bytes::from("read"), Bytes::from("chunks")]
        );
    }
}
