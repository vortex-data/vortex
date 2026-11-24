// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::ByteBuffer;
    use vortex_io::segments::SharedSegmentSource;

    use crate::segments::CountingSegmentSource;
    use crate::segments::SegmentId;
    use crate::segments::SegmentSink;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;

    #[tokio::test]
    async fn test_shared_source_deduplicates_concurrent_requests() {
        let segments = Arc::new(TestSegments::default());
        let source = CountingSegmentSource::new(Arc::clone(&segments) as _);

        // Add a segment to the test source
        let data = ByteBuffer::from(vec![1, 2, 3, 4]);
        let seq_id = SequenceId::root().downgrade();
        segments.write(seq_id, vec![data.clone()]).await.unwrap();

        let shared_source = SharedSegmentSource::new(source.clone());

        // Request the same segment twice concurrently
        let id = SegmentId::from(0);
        let future1 = shared_source.request(id);
        let future2 = shared_source.request(id);

        // Both futures should resolve to the same data
        let (result1, result2) = futures::join!(future1, future2);
        assert_eq!(result1.unwrap().unwrap_host(), data);
        assert_eq!(result2.unwrap().unwrap_host(), data);

        // The inner source should have been called only once
        assert_eq!(source.requests(), 1);
    }

    #[tokio::test]
    async fn test_shared_source_handles_dropped_futures() {
        let segments = Arc::new(TestSegments::default());
        let source = CountingSegmentSource::new(Arc::clone(&segments) as _);

        // Add a segment
        let data = ByteBuffer::from(vec![5, 6, 7, 8]);
        let seq_id = SequenceId::root().downgrade();
        segments.write(seq_id, vec![data.clone()]).await.unwrap();

        let shared_source = SharedSegmentSource::new(source.clone());
        let id = SegmentId::from(0);

        // Create and immediately drop a future
        {
            let _future = shared_source.request(id);
            // Future is dropped here
        }

        // A new request should still work correctly
        let result = shared_source.request(id).await;
        assert_eq!(result.unwrap().unwrap_host(), data);

        // Should have made 2 requests since the first was dropped before completion
        assert_eq!(source.requests(), 2);
    }
}
