// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;

use crate::scan::metrics::ScanUsage;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// A [`SegmentSource`] that adds each completed segment read to a [`ScanUsage`].
///
/// It counts every read that a reader requests through it, also when a cache below it serves the
/// read. Put it above the caches to count requested bytes, and give each scope its own instance:
/// the inner source and its caches stay shared.
pub struct UsageSegmentSource {
    inner: Arc<dyn SegmentSource>,
    usage: Arc<ScanUsage>,
}

impl UsageSegmentSource {
    /// Counts the reads through `inner` in `usage`.
    pub fn new(inner: Arc<dyn SegmentSource>, usage: Arc<ScanUsage>) -> Self {
        Self { inner, usage }
    }
}

impl SegmentSource for UsageSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let usage = Arc::clone(&self.usage);

        self.inner
            .request(id)
            .map(move |segment| {
                if let Ok(buffer) = &segment {
                    usage.record_segment(buffer.len());
                }

                segment
            })
            .boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;

    use super::UsageSegmentSource;
    use crate::scan::metrics::ScanUsage;
    use crate::segments::SegmentId;
    use crate::segments::SegmentSink;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;

    #[tokio::test]
    async fn counts_completed_reads_per_scope() -> VortexResult<()> {
        let segments = TestSegments::default();
        let small = segments
            .write(
                SequenceId::root().downgrade(),
                vec![ByteBuffer::from(vec![0u8; 3])],
            )
            .await?;
        let large = segments
            .write(
                SequenceId::root().downgrade(),
                vec![ByteBuffer::from(vec![0u8; 10])],
            )
            .await?;
        let shared: Arc<dyn SegmentSource> = Arc::new(segments);

        let first = Arc::new(ScanUsage::default());
        let second = Arc::new(ScanUsage::default());
        let first_source = UsageSegmentSource::new(Arc::clone(&shared), Arc::clone(&first));
        let second_source = UsageSegmentSource::new(shared, Arc::clone(&second));

        first_source.request(small).await?;
        first_source.request(large).await?;
        first_source.request(small).await?;
        second_source.request(large).await?;
        assert!(second_source.request(SegmentId::from(99)).await.is_err());

        let first = first.snapshot();
        assert_eq!((first.segments_requested, first.bytes_requested), (3, 16));

        let second = second.snapshot();
        assert_eq!((second.segments_requested, second.bytes_requested), (1, 10));
        Ok(())
    }
}
