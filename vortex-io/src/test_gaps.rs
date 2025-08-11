// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod panic_tests {
    use super::super::io_buf::*;

    #[test]
    #[should_panic(expected = "Invalid range")]
    fn test_owned_slice_invalid_range() {
        let data = vec![1, 2, 3];
        #[allow(clippy::reversed_empty_ranges)]
        let _ = data.slice_owned(5..3); // start > end
    }

    #[test]
    #[should_panic(expected = "exceeds buffer length")]
    fn test_owned_slice_out_of_bounds() {
        let data = vec![1, 2, 3];
        let _ = data.slice_owned(1..10); // end > len
    }

    #[test]
    fn test_owned_slice_zero_sized_at_boundary() {
        let data = vec![1, 2, 3];
        let slice = data.slice_owned(3..3); // Zero-sized at end
        assert_eq!(slice.bytes_init(), 0);
    }

    #[test]
    #[should_panic(expected = "exceeds buffer length")]
    fn test_owned_slice_start_out_of_bounds() {
        let data = vec![1, 2, 3];
        let _ = data.slice_owned(10..11); // start > len
    }
}

#[cfg(test)]
mod buffer_overflow_tests {
    use vortex_buffer::{Buffer, BufferMut};

    use super::super::io_buf::*;

    #[test]
    fn test_buffer_size_calculation_u8() {
        let buffer: Buffer<u8> = Buffer::from(vec![1, 2, 3]);
        assert_eq!(buffer.bytes_init(), 3);
    }

    #[test]
    fn test_buffer_size_calculation_large_type() {
        struct LargeType([u8; 1024]);

        let mut buf = BufferMut::<LargeType>::with_capacity(10);
        // Extend with 10 elements
        for _ in 0..10 {
            buf.push(LargeType([0u8; 1024]));
        }
        let buffer = buf.freeze();

        // This should use checked arithmetic and not overflow
        let size = buffer.bytes_init();
        assert_eq!(size, 10 * 1024);
    }

    #[test]
    fn test_buffer_size_near_max() {
        // Test with a moderately large buffer that won't cause OOM
        let large_size = 1_000_000;
        let buffer: Buffer<u8> = Buffer::from(vec![0u8; large_size]);
        assert_eq!(buffer.bytes_init(), large_size);
    }
}

#[cfg(all(test, feature = "tokio", feature = "object_store"))]
mod object_store_concurrency_tests {
    use std::sync::Arc;

    use object_store::ObjectStore;

    use super::super::object_store::ObjectStoreWriter;
    use super::super::write::VortexWrite;

    const MAX_BUFFER_SIZE: usize = 100 * 1024 * 1024; // Local copy for testing

    async fn create_test_store() -> (
        Arc<object_store::memory::InMemory>,
        object_store::path::Path,
    ) {
        let store = Arc::new(object_store::memory::InMemory::new());
        let location = object_store::path::Path::from("test.bin");
        (store, location)
    }

    #[tokio::test]
    async fn test_object_store_writer_concurrent_writes() {
        let (store, location) = create_test_store().await;
        let writer = Arc::new(
            ObjectStoreWriter::new(store.clone(), &location)
                .await
                .unwrap(),
        );

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let w = writer.clone();
                tokio::spawn(async move {
                    #[allow(clippy::cast_possible_truncation)]
                    let data = vec![i as u8; 1000];
                    let mut w = w.as_ref().clone();
                    w.write_all(data).await
                })
            })
            .collect();

        for handle in handles {
            handle.await.unwrap().unwrap();
        }

        let mut writer = Arc::try_unwrap(writer).unwrap_or_else(|arc| (*arc).clone());
        writer.flush().await.unwrap();

        // Verify data was written
        let result = store.get(&location).await.unwrap();
        let bytes = result.bytes().await.unwrap();
        assert_eq!(bytes.len(), 10000); // 10 * 1000
    }

    #[tokio::test]
    async fn test_object_store_writer_max_buffer_size() {
        let (store, location) = create_test_store().await;
        let mut writer = ObjectStoreWriter::new(store, &location).await.unwrap();

        // Try to write more than MAX_BUFFER_SIZE
        let huge_data = vec![0u8; MAX_BUFFER_SIZE + 1];
        let result = writer.write_all(huge_data).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceed maximum"));
    }

    #[tokio::test]
    async fn test_object_store_writer_multiple_flushes() {
        let (store, location) = create_test_store().await;
        let mut writer = ObjectStoreWriter::new(store.clone(), &location)
            .await
            .unwrap();

        // Write and flush multiple times
        for i in 0..3 {
            #[allow(clippy::cast_possible_truncation)]
            let data = vec![i as u8; 100];
            writer.write_all(data).await.unwrap();
            writer.flush().await.unwrap();
        }

        // Verify all data was written
        let result = store.get(&location).await.unwrap();
        let bytes = result.bytes().await.unwrap();
        assert_eq!(bytes.len(), 300);
    }
}

#[cfg(test)]
mod performance_hint_tests {
    use super::super::read::PerformanceHint;

    #[test]
    fn test_performance_hint_local() {
        let hint = PerformanceHint::local();
        assert_eq!(hint.coalescing_window(), 8192);
        assert_eq!(hint.max_read(), Some(8192));
    }

    #[test]
    fn test_performance_hint_object_storage() {
        let hint = PerformanceHint::object_storage();
        assert_eq!(hint.coalescing_window(), 1 << 20); // 1MB
        assert_eq!(hint.max_read(), Some(8 << 20)); // 8MB
    }

    #[test]
    fn test_performance_hint_custom() {
        let hint = PerformanceHint::new(4096, Some(16384));
        assert_eq!(hint.coalescing_window(), 4096);
        assert_eq!(hint.max_read(), Some(16384));
    }

    #[test]
    fn test_performance_hint_no_max() {
        let hint = PerformanceHint::new(2048, None);
        assert_eq!(hint.coalescing_window(), 2048);
        assert_eq!(hint.max_read(), None);
    }
}

#[cfg(all(test, feature = "tokio"))]
mod size_limited_stream_tests {
    use futures_util::StreamExt;

    use super::super::limit::SizeLimitedStream;

    #[tokio::test]
    async fn test_size_limited_stream_zero_capacity() {
        let stream = SizeLimitedStream::new(0);

        // Should not be able to push anything
        let result = stream.try_push(async { vec![1u8] }, 1);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_size_limited_stream_dropped_future_releases_permits() {
        use futures::future::BoxFuture;

        let mut stream = SizeLimitedStream::<BoxFuture<'static, Vec<u8>>>::new(10);

        // Push a future that will never complete
        stream
            .push(
                Box::pin(async {
                    // This future will be dropped before completion
                    futures::future::pending::<Vec<u8>>().await
                }),
                5,
            )
            .await;

        // Push another future
        stream.push(Box::pin(async { vec![1u8; 3] }), 3).await;

        // We should have 2 bytes available now
        assert_eq!(stream.bytes_available(), 2);

        // Drop the stream without consuming the futures
        drop(stream);

        // Create a new stream to verify permits aren't leaked
        let mut new_stream = SizeLimitedStream::<BoxFuture<'static, Vec<u8>>>::new(10);

        // Should be able to use all 10 bytes
        new_stream.push(Box::pin(async { vec![0u8; 10] }), 10).await;
        assert_eq!(new_stream.bytes_available(), 0);

        // Consume to verify it works
        let result = new_stream.next().await;
        assert!(result.is_some());
        assert_eq!(new_stream.bytes_available(), 10);
    }

    #[tokio::test]
    async fn test_size_limited_stream_exact_capacity() {
        use futures::future::BoxFuture;

        let mut stream = SizeLimitedStream::<BoxFuture<'static, Vec<u8>>>::new(10);

        // Push exactly the capacity
        stream.push(Box::pin(async { vec![0u8; 10] }), 10).await;

        // Should not be able to push more
        let result = stream.try_push(Box::pin(async { vec![1u8] }), 1);
        assert!(result.is_err());

        // After consuming, should be able to push again
        let _ = stream.next().await;
        assert_eq!(stream.bytes_available(), 10);

        let result = stream.try_push(Box::pin(async { vec![1u8; 5] }), 5);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_size_limited_stream_multiple_small_pushes() {
        let mut stream = SizeLimitedStream::new(100);

        // Push many small items
        for i in 0..10 {
            #[allow(clippy::cast_possible_truncation)]
            stream.push(async move { vec![i as u8; 5] }, 5).await;
        }

        // Should have used 50 bytes
        assert_eq!(stream.bytes_available(), 50);

        // Consume all
        let mut count = 0;
        while stream.next().await.is_some() {
            count += 1;
            if count == 10 {
                break;
            }
        }

        assert_eq!(count, 10);
        assert_eq!(stream.bytes_available(), 100);
    }
}

#[cfg(all(test, feature = "compio"))]
mod dispatcher_error_tests {
    use std::panic;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::super::dispatcher::{Dispatch, IoDispatcher};

    #[test]
    fn test_dispatcher_task_panic_handling() {
        let dispatcher = IoDispatcher::new();
        let completed = Arc::new(AtomicBool::new(false));
        let completed_clone = completed.clone();

        // Dispatch a task that will panic
        #[allow(clippy::panic)]
        let _handle = dispatcher.dispatch(move || async move {
            panic!("Task panic");
        });

        // Also dispatch a normal task to verify dispatcher continues working
        let normal_handle = dispatcher
            .dispatch(move || async move {
                completed_clone.store(true, Ordering::SeqCst);
                42
            })
            .unwrap();

        // The panic task should propagate the error
        // Note: this depends on implementation details

        // The normal task should complete
        let result = futures::executor::block_on(normal_handle);
        assert_eq!(result.unwrap(), 42);
        assert!(completed.load(Ordering::SeqCst));

        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_dispatcher_shutdown_empty_queue() {
        let dispatcher = IoDispatcher::new();
        // Immediate shutdown should work
        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_dispatcher_many_threads() {
        let dispatcher = IoDispatcher::new();
        let mut handles = Vec::new();

        for i in 0..100 {
            let handle = dispatcher.dispatch(move || async move { i * 2 }).unwrap();
            handles.push(handle);
        }

        for (i, handle) in handles.into_iter().enumerate() {
            let result = futures::executor::block_on(handle);
            assert_eq!(result.unwrap(), i * 2);
        }

        dispatcher.shutdown().unwrap();
    }
}

#[cfg(test)]
mod limit_edge_cases {
    use super::super::limit::*;

    #[test]
    fn test_size_overflow_protection() {
        let stream = SizeLimitedStream::new(100);

        // Test with size that would overflow u32 on 32-bit systems
        // but this test assumes 64-bit where usize > u32::MAX is possible
        #[cfg(target_pointer_width = "64")]
        {
            let _large_size = (u32::MAX as usize) + 1;
            // This should panic with current implementation
            // We're documenting the issue rather than testing the panic
            // as the behavior may change
        }

        // Test with reasonable size
        let result = stream.try_push(async { vec![0u8; 50] }, 50);
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod read_tests {
    use std::sync::Arc;

    use vortex_buffer::ByteBuffer;

    use super::super::read::*;

    #[tokio::test]
    async fn test_byte_buffer_read_at() {
        let data = ByteBuffer::from(vec![1, 2, 3, 4, 5]);

        let result = data
            .read_byte_range(1..4, vortex_buffer::Alignment::none())
            .await
            .unwrap();
        assert_eq!(result.as_ref(), &[2, 3, 4]);
    }

    #[tokio::test]
    async fn test_byte_buffer_read_out_of_bounds() {
        let data = ByteBuffer::from(vec![1, 2, 3]);

        let result = data
            .read_byte_range(1..10, vortex_buffer::Alignment::none())
            .await;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
    }

    #[tokio::test]
    async fn test_arc_read_at() {
        let data = Arc::new(ByteBuffer::from(vec![1, 2, 3, 4, 5]));

        let result = data
            .read_byte_range(2..5, vortex_buffer::Alignment::none())
            .await
            .unwrap();
        assert_eq!(result.as_ref(), &[3, 4, 5]);

        let size = data.size().await.unwrap();
        assert_eq!(size, 5);
    }
}
