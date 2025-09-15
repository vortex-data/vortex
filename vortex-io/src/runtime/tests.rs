// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(feature = "tokio")]
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use tempfile::NamedTempFile;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::VortexResult;

use crate::VortexReadAt;
use crate::file::{IntoReadSource, IoRequest, ReadSource, ReadSourceRef};
use crate::runtime::Handle;
use crate::runtime::single::block_on;
use crate::runtime::tokio::TokioRuntime;

// Test data
const TEST_DATA: &[u8] = b"Hello, World! This is test data for FileRead.";
const TEST_OFFSET: u64 = 7;
const TEST_LEN: usize = 5;

// ============================================================================
// Basic FileRead tests with in-memory buffer
// ============================================================================

#[test]
fn test_file_read_with_single_thread_runtime() {
    let result = block_on(|handle| {
        async move {
            let buffer = ByteBuffer::from(TEST_DATA.to_vec());
            let file_read = handle.open_read(buffer).unwrap();

            // Read a slice
            let result = file_read
                .read_at(TEST_OFFSET, TEST_LEN, Alignment::new(1))
                .await
                .unwrap();
            assert_eq!(
                result.as_slice(),
                &TEST_DATA[TEST_OFFSET as usize..][..TEST_LEN]
            );

            // Read the entire file
            let full = file_read
                .read_at(0, TEST_DATA.len(), Alignment::new(1))
                .await
                .unwrap();
            assert_eq!(full.as_slice(), TEST_DATA);

            "success"
        }
        .boxed_local()
    });
    assert_eq!(result, "success");
}

#[tokio::test]
async fn test_file_read_with_tokio_runtime() {
    let handle = TokioRuntime::current();
    let buffer = ByteBuffer::from(TEST_DATA.to_vec());
    let file_read = handle.open_read(buffer).unwrap();

    // Read a slice
    let result = file_read
        .read_at(TEST_OFFSET, TEST_LEN, Alignment::new(1))
        .await
        .unwrap();
    assert_eq!(
        result.as_slice(),
        &TEST_DATA[TEST_OFFSET as usize..][..TEST_LEN]
    );

    // Read the entire file
    let full = file_read
        .read_at(0, TEST_DATA.len(), Alignment::new(1))
        .await
        .unwrap();
    assert_eq!(full.as_slice(), TEST_DATA);
}

// ============================================================================
// Test with actual files
// ============================================================================

#[test]
fn test_file_read_with_real_file_single_thread() {
    use std::io::Write;

    let result = block_on(|handle| {
        async move {
            // Create a temporary file
            let mut temp_file = NamedTempFile::new().unwrap();
            temp_file.write_all(TEST_DATA).unwrap();
            temp_file.flush().unwrap();

            // Open and read the file
            let file_read = handle.open_read(temp_file.path()).unwrap();

            // Read a slice
            let result = file_read
                .read_at(TEST_OFFSET, TEST_LEN, Alignment::new(1))
                .await
                .unwrap();
            assert_eq!(
                result.as_slice(),
                &TEST_DATA[TEST_OFFSET as usize..][..TEST_LEN]
            );

            // Read the entire file
            let full = file_read
                .read_at(0, TEST_DATA.len(), Alignment::new(1))
                .await
                .unwrap();
            assert_eq!(full.as_slice(), TEST_DATA);

            "success"
        }
        .boxed_local()
    });
    assert_eq!(result, "success");
}

#[tokio::test]
async fn test_file_read_with_real_file_tokio() {
    use std::io::Write;

    // Create a temporary file
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(TEST_DATA).unwrap();
    temp_file.flush().unwrap();

    let handle = TokioRuntime::current();
    let file_read = handle.open_read(temp_file.path()).unwrap();

    // Read a slice
    let result = file_read
        .read_at(TEST_OFFSET, TEST_LEN, Alignment::new(1))
        .await
        .unwrap();
    assert_eq!(
        result.as_slice(),
        &TEST_DATA[TEST_OFFSET as usize..][..TEST_LEN]
    );

    // Read the entire file
    let full = file_read
        .read_at(0, TEST_DATA.len(), Alignment::new(1))
        .await
        .unwrap();
    assert_eq!(full.as_slice(), TEST_DATA);
}

// ============================================================================
// Test concurrent reads
// ============================================================================

#[tokio::test]
async fn test_concurrent_reads() {
    let handle = TokioRuntime::current();
    let buffer = ByteBuffer::from(TEST_DATA.to_vec());
    let file_read = handle.open_read(buffer).unwrap();

    // Issue multiple concurrent reads
    let futures = vec![
        file_read.read_at(0, 5, Alignment::new(1)),
        file_read.read_at(5, 5, Alignment::new(1)),
        file_read.read_at(10, 5, Alignment::new(1)),
        file_read.read_at(15, 5, Alignment::new(1)),
    ];

    let results = futures::future::join_all(futures).await;

    assert_eq!(results[0].as_ref().unwrap().as_slice(), &TEST_DATA[0..5]);
    assert_eq!(results[1].as_ref().unwrap().as_slice(), &TEST_DATA[5..10]);
    assert_eq!(results[2].as_ref().unwrap().as_slice(), &TEST_DATA[10..15]);
    assert_eq!(results[3].as_ref().unwrap().as_slice(), &TEST_DATA[15..20]);
}

// ============================================================================
// Test Handle spawn methods
// ============================================================================

#[test]
fn test_handle_spawn_future() {
    let result = block_on(|handle| {
        async move {
            let task = handle.spawn(async move { 42 });
            task.await
        }
        .boxed_local()
    });
    assert_eq!(result, 42);
}

#[tokio::test]
async fn test_handle_spawn_cpu() {
    let handle = TokioRuntime::current();
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();

    let task = handle.spawn_cpu(move || {
        c.fetch_add(1, Ordering::SeqCst);
        100
    });

    let result = task.await;
    assert_eq!(result, 100);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

// ============================================================================
// Test custom IoSource implementation
// ============================================================================

struct CountingIoSource {
    data: ByteBuffer,
    read_count: Arc<AtomicUsize>,
}

impl ReadSource for CountingIoSource {
    fn uri(&self) -> &Arc<str> {
        static URI: std::sync::LazyLock<Arc<str>> =
            std::sync::LazyLock::new(|| Arc::from("counting://test"));
        &URI
    }

    fn coalesce_window(&self) -> Option<crate::file::CoalesceWindow> {
        None
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let len = self.data.len() as u64;
        async move { Ok(len) }.boxed()
    }

    fn drive_send(
        self: Arc<Self>,
        mut requests: BoxStream<'static, IoRequest>,
    ) -> BoxFuture<'static, ()> {
        async move {
            while let Some(req) = requests.next().await {
                self.read_count.fetch_add(1, Ordering::SeqCst);

                let offset = req.offset() as usize;
                let len = req.len();

                let result = if offset + len > self.data.len() {
                    Err(vortex_error::vortex_err!("Read out of bounds"))
                } else {
                    let mut buffer = ByteBufferMut::with_capacity_aligned(len, req.alignment());
                    unsafe { buffer.set_len(len) };
                    buffer
                        .as_mut_slice()
                        .copy_from_slice(&self.data.as_slice()[offset..offset + len]);
                    Ok(buffer.freeze())
                };
                req.resolve(result);
            }
        }
        .boxed()
    }
}

impl IntoReadSource for CountingIoSource {
    fn into_read_source(self, _handle: Handle) -> VortexResult<ReadSourceRef> {
        Ok(Arc::new(self))
    }
}

#[tokio::test]
async fn test_custom_io_source() {
    let handle = TokioRuntime::current();
    let read_count = Arc::new(AtomicUsize::new(0));

    let source = CountingIoSource {
        data: ByteBuffer::from(TEST_DATA.to_vec()),
        read_count: read_count.clone(),
    };

    let file_read = handle.open_read(source).unwrap();

    // Perform several reads
    let _ = file_read.read_at(0, 5, Alignment::new(1)).await.unwrap();
    let _ = file_read.read_at(5, 5, Alignment::new(1)).await.unwrap();
    let _ = file_read.read_at(10, 5, Alignment::new(1)).await.unwrap();

    // Check that our custom IoSource was called 3 times
    assert_eq!(read_count.load(Ordering::SeqCst), 3);
}

// ============================================================================
// Test error handling
// ============================================================================

#[tokio::test]
async fn test_read_out_of_bounds() {
    let handle = TokioRuntime::current();
    let buffer = ByteBuffer::from(TEST_DATA.to_vec());
    let file_read = handle.open_read(buffer).unwrap();

    // Try to read beyond the buffer
    let result = file_read.read_at(100, 10, Alignment::new(1)).await;
    assert!(result.is_err());

    // Try to read with length that exceeds buffer
    let result = file_read.read_at(40, 20, Alignment::new(1)).await;
    assert!(result.is_err());
}

// ============================================================================
// Test task detaching
// ============================================================================

#[tokio::test]
async fn test_task_detach() {
    let handle = TokioRuntime::current();
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();

    let task = handle.spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        c.fetch_add(1, Ordering::SeqCst);
        42
    });

    // Detach the task so it continues running
    task.detach();

    // Wait for task to complete
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    // Task should have completed even though we detached it
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

// ============================================================================
// Test nested spawns
// ============================================================================

#[test]
fn test_nested_spawns() {
    let result =
        block_on(|h| h.spawn_nested(|h| async move { h.spawn(async move { 42 }).await + 10 }));
    assert_eq!(result, 52);
}
