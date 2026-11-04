// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(feature = "tokio")]
#![expect(clippy::cast_possible_truncation)]

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::task::Poll;
use std::time::Instant;

use futures::FutureExt;
use futures::channel::oneshot;
use rstest::rstest;
use tempfile::NamedTempFile;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;

use crate::VortexReadAt;
use crate::runtime::Handle;
use crate::runtime::JoinOutcome;
use crate::runtime::Task;
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

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic>")
}

// Joining a spawned future that panicked must re-raise the *original* panic, not a generic
// "Runtime dropped task" message.
#[test]
fn test_spawned_future_panic_reraises_original() {
    let outcome = block_on(|handle| {
        async move {
            let task: Task<()> = handle.spawn(async move { panic!("original spawn panic") });
            AssertUnwindSafe(task).catch_unwind().await
        }
        .boxed_local()
    });

    let payload = outcome.expect_err("panicking task must propagate a panic");
    assert!(
        panic_message(&*payload).contains("original spawn panic"),
        "got: {:?}",
        panic_message(&*payload)
    );
}

// Same for CPU tasks.
#[tokio::test]
async fn test_spawned_cpu_task_panic_reraises_original() {
    let handle = TokioRuntime::current();
    let task: Task<()> = handle.spawn_cpu(|| panic!("original cpu panic"));

    let payload = AssertUnwindSafe(task)
        .catch_unwind()
        .await
        .expect_err("panicking cpu task must propagate a panic");
    assert!(
        panic_message(&*payload).contains("original cpu panic"),
        "got: {:?}",
        panic_message(&*payload)
    );
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

// ============================================================================
// Test joining a task while the runtime shuts down
// ============================================================================

/// Sets its flag when dropped, so a test can observe the runtime dropping a spawned future.
struct SetOnDrop(Arc<AtomicBool>);

impl Drop for SetOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

type SpawnPending = fn(&Handle, SetOnDrop) -> Task<()>;

fn spawn_pending(handle: &Handle, guard: SetOnDrop) -> Task<()> {
    handle.spawn(async move {
        let _guard = guard;
        futures::future::pending::<()>().await
    })
}

fn spawn_io_pending(handle: &Handle, guard: SetOnDrop) -> Task<()> {
    handle.spawn_io(async move {
        let _guard = guard;
        futures::future::pending::<()>().await
    })
}

fn spawn_nested_pending(handle: &Handle, guard: SetOnDrop) -> Task<()> {
    handle.spawn_nested(|_| async move {
        let _guard = guard;
        futures::future::pending::<()>().await
    })
}

/// How a detached task saw its [`Task`] once the runtime had dropped the task's future.
enum ShutdownJoin {
    /// Polling the task through its `Future` impl panicked with this message.
    Panicked(String),
    /// Polling the task through its `Future` impl returned without panicking.
    Polled(Poll<()>),
    /// `Task::poll_join` reported this outcome.
    Joined(JoinOutcome<()>),
}

/// Spawns `spawn_task` from a detached Tokio task (like a DataFusion stream task), then drops the
/// runtime while that detached task is still inside a single poll. Once another worker has dropped
/// the spawned task's future, the detached task joins its [`Task`] and reports what it saw.
fn join_during_runtime_shutdown(
    worker_threads: usize,
    spawn_task: SpawnPending,
    use_poll_join: bool,
) -> ShutdownJoin {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()
        .expect("build Tokio runtime");
    let dropped = Arc::new(AtomicBool::new(false));
    let (outcome_tx, outcome_rx) = mpsc::channel();

    rt.block_on({
        let dropped = Arc::clone(&dropped);
        async move {
            let (started_tx, started_rx) = oneshot::channel();
            drop(tokio::spawn(async move {
                let mut task =
                    spawn_task(&TokioRuntime::current(), SetOnDrop(Arc::clone(&dropped)));
                started_tx
                    .send(())
                    .expect("test is waiting for the task to start");
                // Stay inside this poll until shutdown drops the spawned future, so the runtime
                // cannot drop this task first.
                let deadline = Instant::now() + std::time::Duration::from_secs(10);
                while !dropped.load(Ordering::SeqCst) {
                    assert!(
                        Instant::now() < deadline,
                        "runtime shutdown never dropped the spawned task"
                    );
                    std::hint::spin_loop();
                }
                // The guard drops just before the task's result sender, so spin until the result
                // channel reports the task's future gone.
                let joined = loop {
                    let polled =
                        futures::future::poll_fn(|cx| Poll::Ready(task.poll_join(cx))).await;
                    if let Poll::Ready(outcome) = polled {
                        break outcome;
                    }
                    assert!(
                        Instant::now() < deadline,
                        "the dropped task's result channel never closed"
                    );
                    std::hint::spin_loop();
                };
                let outcome = if use_poll_join {
                    ShutdownJoin::Joined(joined)
                } else {
                    // A closed channel keeps reporting closed, so this poll sees the same teardown.
                    let polled = AssertUnwindSafe(futures::future::poll_fn(|cx| {
                        Poll::Ready(task.poll_unpin(cx))
                    }))
                    .catch_unwind()
                    .await;
                    match polled {
                        Ok(poll) => ShutdownJoin::Polled(poll),
                        Err(panic) => ShutdownJoin::Panicked(panic_message(&*panic).to_owned()),
                    }
                };
                drop(outcome_tx.send(outcome));
            }));
            started_rx.await.expect("detached task started");
        }
    });
    drop(rt);

    outcome_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("detached task reported its join outcome")
}

// Awaiting a task whose future the runtime dropped during shutdown is a benign teardown, not a
// runtime bug: it must not panic the awaiting task.
#[rstest]
fn test_task_polled_during_runtime_shutdown_does_not_panic(
    #[values(2, 4)] worker_threads: usize,
    #[values(
        spawn_pending as SpawnPending,
        spawn_io_pending as SpawnPending,
        spawn_nested_pending as SpawnPending
    )]
    spawn_task: SpawnPending,
) {
    match join_during_runtime_shutdown(worker_threads, spawn_task, false) {
        ShutdownJoin::Panicked(message) => {
            panic!("awaiting a task during runtime shutdown panicked: {message}")
        }
        ShutdownJoin::Polled(poll) => assert!(poll.is_pending()),
        ShutdownJoin::Joined(_) => unreachable!("joined through the Future impl"),
    }
}

// `Task::poll_join` reports the same teardown as `Aborted`.
#[rstest]
fn test_task_poll_join_during_runtime_shutdown_reports_aborted(
    #[values(2, 4)] worker_threads: usize,
    #[values(
        spawn_pending as SpawnPending,
        spawn_io_pending as SpawnPending,
        spawn_nested_pending as SpawnPending
    )]
    spawn_task: SpawnPending,
) {
    assert!(matches!(
        join_during_runtime_shutdown(worker_threads, spawn_task, true),
        ShutdownJoin::Joined(JoinOutcome::Aborted)
    ));
}
