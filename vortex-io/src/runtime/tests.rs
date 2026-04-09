// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(feature = "tokio")]
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::future::BoxFuture;
use tempfile::NamedTempFile;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

use crate::VortexReadAt;
use crate::runtime::single::block_on;
use crate::runtime::tokio::TokioRuntime;
use crate::std_file::FileReadAt;

// Test data
const TEST_DATA: &[u8] = b"Hello, World! This is test data for FileRead.";
const TEST_OFFSET: u64 = 7;
const TEST_LEN: usize = 5;

// ============================================================================
// Basic FileRead tests with in-memory buffer
// ============================================================================

#[test]
fn test_file_read_with_single_thread_runtime() {
    let result = block_on(|_handle| {
        async move {
            let file_read: Arc<dyn VortexReadAt> = Arc::new(ByteBuffer::from(TEST_DATA.to_vec()));

            // Read a slice
            let result = file_read
                .read_at(TEST_OFFSET, TEST_LEN, Alignment::new(1))
                .await
                .unwrap();
            assert_eq!(
                result.to_host().await.as_slice(),
                &TEST_DATA[TEST_OFFSET as usize..][..TEST_LEN]
            );

            // Read the entire file
            let full = file_read
                .read_at(0, TEST_DATA.len(), Alignment::new(1))
                .await
                .unwrap();
            assert_eq!(full.to_host().await.as_slice(), TEST_DATA);

            "success"
        }
        .boxed_local()
    });
    assert_eq!(result, "success");
}

#[tokio::test]
async fn test_file_read_with_tokio_runtime() {
    let file_read: Arc<dyn VortexReadAt> = Arc::new(ByteBuffer::from(TEST_DATA.to_vec()));

    // Read a slice
    let result = file_read
        .read_at(TEST_OFFSET, TEST_LEN, Alignment::new(1))
        .await
        .unwrap();
    assert_eq!(
        result.to_host().await.as_slice(),
        &TEST_DATA[TEST_OFFSET as usize..][..TEST_LEN]
    );

    // Read the entire file
    let full = file_read
        .read_at(0, TEST_DATA.len(), Alignment::new(1))
        .await
        .unwrap();
    assert_eq!(full.to_host().await.as_slice(), TEST_DATA);
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
            let file_read: Arc<dyn VortexReadAt> =
                Arc::new(FileReadAt::open(temp_file.path(), handle.clone()).unwrap());

            // Read a slice
            let result = file_read
                .read_at(TEST_OFFSET, TEST_LEN, Alignment::new(1))
                .await
                .unwrap();
            assert_eq!(
                result.to_host().await.as_slice(),
                &TEST_DATA[TEST_OFFSET as usize..][..TEST_LEN]
            );

            // Read the entire file
            let full = file_read
                .read_at(0, TEST_DATA.len(), Alignment::new(1))
                .await
                .unwrap();
            assert_eq!(full.to_host().await.as_slice(), TEST_DATA);

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
    let file_read: Arc<dyn VortexReadAt> =
        Arc::new(FileReadAt::open(temp_file.path(), handle.clone()).unwrap());

    // Read a slice
    let result = file_read
        .read_at(TEST_OFFSET, TEST_LEN, Alignment::new(1))
        .await
        .unwrap();
    assert_eq!(
        result.to_host().await.as_slice(),
        &TEST_DATA[TEST_OFFSET as usize..][..TEST_LEN]
    );

    // Read the entire file
    let full = file_read
        .read_at(0, TEST_DATA.len(), Alignment::new(1))
        .await
        .unwrap();
    assert_eq!(full.to_host().await.as_slice(), TEST_DATA);
}

// ============================================================================
// Test concurrent reads
// ============================================================================

#[tokio::test]
async fn test_concurrent_reads() {
    let read_at: Arc<dyn VortexReadAt> = Arc::new(ByteBuffer::from(TEST_DATA.to_vec()));

    // Issue multiple concurrent reads
    let futures = vec![
        read_at.read_at(0, 5, Alignment::new(1)),
        read_at.read_at(5, 5, Alignment::new(1)),
        read_at.read_at(10, 5, Alignment::new(1)),
        read_at.read_at(15, 5, Alignment::new(1)),
    ];

    let results = futures::future::join_all(futures).await;

    assert_eq!(
        results[0].as_ref().unwrap().to_host().await.as_slice(),
        &TEST_DATA[0..5]
    );
    assert_eq!(
        results[1].as_ref().unwrap().to_host().await.as_slice(),
        &TEST_DATA[5..10]
    );
    assert_eq!(
        results[2].as_ref().unwrap().to_host().await.as_slice(),
        &TEST_DATA[10..15]
    );
    assert_eq!(
        results[3].as_ref().unwrap().to_host().await.as_slice(),
        &TEST_DATA[15..20]
    );
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
    let c = Arc::clone(&counter);

    let task = handle.spawn_cpu(move || {
        c.fetch_add(1, Ordering::SeqCst);
        100
    });

    let result = task.await;
    assert_eq!(result, 100);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

// ============================================================================
// Test custom VortexRead implementation
// ============================================================================

struct CountingReadAt {
    data: ByteBuffer,
    read_count: Arc<AtomicUsize>,
}

impl VortexReadAt for CountingReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        None
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let len = self.data.len() as u64;
        async move { Ok(len) }.boxed()
    }

    fn concurrency(&self) -> usize {
        16
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.read_count.fetch_add(1, Ordering::SeqCst);
        let data = self.data.clone();
        async move {
            let start = offset as usize;
            if start + length > data.len() {
                return Err(vortex_error::vortex_err!("Read out of bounds"));
            }
            let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
            unsafe { buffer.set_len(length) };
            buffer
                .as_mut_slice()
                .copy_from_slice(&data.as_slice()[start..start + length]);
            Ok(BufferHandle::new_host(buffer.freeze()))
        }
        .boxed()
    }
}

#[tokio::test]
async fn test_custom_vortex_read() {
    let read_count = Arc::new(AtomicUsize::new(0));

    let read_at: Arc<dyn VortexReadAt> = Arc::new(CountingReadAt {
        data: ByteBuffer::from(TEST_DATA.to_vec()),
        read_count: Arc::clone(&read_count),
    });

    // Perform several reads
    read_at.read_at(0, 5, Alignment::new(1)).await.unwrap();
    read_at.read_at(5, 5, Alignment::new(1)).await.unwrap();
    read_at.read_at(10, 5, Alignment::new(1)).await.unwrap();

    // Check that our custom VortexRead was called 3 times
    assert_eq!(read_count.load(Ordering::SeqCst), 3);
}

// ============================================================================
// Test error handling
// ============================================================================

#[tokio::test]
async fn test_read_out_of_bounds() {
    let reader: Arc<dyn VortexReadAt> = Arc::new(ByteBuffer::from(TEST_DATA.to_vec()));

    // Try to read beyond the buffer
    let result = reader.read_at(100, 10, Alignment::new(1)).await;
    assert!(result.is_err());

    // Try to read with length that exceeds buffer
    let result = reader.read_at(40, 20, Alignment::new(1)).await;
    assert!(result.is_err());
}

// ============================================================================
// Test task detaching
// ============================================================================

#[tokio::test]
async fn test_task_detach() {
    let handle = TokioRuntime::current();
    let counter = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&counter);
    let (tx, rx) = oneshot::channel::<()>();

    let task = handle.spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        c.fetch_add(1, Ordering::SeqCst);
        tx.send(())
    });

    // Detach the task so it continues running
    task.detach();

    // Wait for task to complete
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let _ = rx.await;

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
