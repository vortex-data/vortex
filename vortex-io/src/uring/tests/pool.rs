// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::poll_fn;
use std::io::Write;
use std::panic::AssertUnwindSafe;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Barrier;
use std::task::Poll;
use std::thread;
use std::thread::ThreadId;
use std::time::Duration;

use futures::FutureExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::channel::oneshot;
use parking_lot::Mutex;
use rstest::rstest;
use tempfile::NamedTempFile;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::TestMode;
use super::request;
use crate::read_at2::ReadId;
use crate::read_at2::ReadResult;
use crate::uring::UringOptions;
use crate::uring::UringWorkerPool;

#[rstest]
#[tokio::test]
async fn idle_worker_releases_cancelled_read_mapping(
    #[values(1, 2)] max_active_jobs: usize,
    #[values(0, 1)] max_control_jobs: usize,
    #[values(false, true)] reuse_across_workers: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[53; 4096])?;
    let file = TestMode::Mapped.open(temporary.path())?;
    let pool = UringWorkerPool::new(
        1,
        UringOptions {
            reuse_across_workers,
            max_active_jobs,
            max_control_jobs,
            ..Default::default()
        },
    )?;
    let mapping = pool
        .spawn(move || {
            let mapped = file
                .0
                .mapping
                .as_ref()
                .expect("mapped file")
                .get(&file.0.file, file.0.size)?
                .expect("nonempty file");
            let mapping = Arc::downgrade(&mapped);
            drop(mapped);
            let reader = file.session()?;
            reader.submit(&[request(0, 0, 4096)])?;
            drop(reader);
            assert!(file.release_mapping());
            Ok(mapping)
        })
        .await?;
    // No later job may be needed to release the cancelled request's mapping.
    let released = tokio::time::timeout(Duration::from_secs(1), async {
        while mapping.strong_count() > 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await;
    assert!(released.is_ok(), "idle worker retained a cancelled mapping");
    Ok(())
}

#[rstest]
#[tokio::test]
async fn another_worker_completes_a_shared_read_after_its_originating_job_returns(
    #[values(1, 2)] max_active_jobs: usize,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[43; 4096])?;
    let file = TestMode::Mapped.open(temporary.path())?;
    let options = UringOptions {
        reuse_across_workers: true,
        max_active_jobs,
        ..Default::default()
    };
    let first_pool = UringWorkerPool::new(1, options)?;
    let second_pool = UringWorkerPool::new(1, options)?;
    let both_submitted = Arc::new(Barrier::new(2));
    let first_barrier = Arc::clone(&both_submitted);
    let (started, start) = oneshot::channel();
    let first_file = file.clone();
    let first = first_pool.spawn(move || {
        let reader = first_file.session()?;
        reader.submit(&[request(0, 0, 4096)])?;
        started.send(()).expect("start receiver");
        first_barrier.wait();
        // The leader's scope cancels here, before its queued read has reached the kernel.
        Ok(thread::current().id())
    });
    start.await.expect("first worker started");
    let second_file = file.clone();
    let second = second_pool.spawn(move || {
        let reader = second_file.session()?;
        reader.submit(&[request(0, 0, 4096)])?;
        both_submitted.wait();
        let ReadResult::Bytes(bytes) = reader.wait()?.result? else {
            panic!("expected bytes")
        };
        assert_eq!(bytes.try_into_host_sync()?.as_slice(), &[43; 4096]);
        Ok(thread::current().id())
    });
    let (first_thread, second_thread) = futures::try_join!(first, second)?;
    assert_ne!(first_thread, second_thread);
    assert_eq!(file.stats().physical_reads, 1);
    assert_eq!(file.0.inflight.get().expect("registry").pending_count(), 0);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn dropping_a_pool_on_a_worker_allows_shared_reads_to_finish(
    #[values(1, 2)] max_active_jobs: usize,
    #[values(false, true)] separate_pools: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[47; 4096])?;
    let file = TestMode::Mapped.open(temporary.path())?;
    let options = UringOptions {
        reuse_across_workers: true,
        max_active_jobs,
        ..Default::default()
    };
    let first_pool = UringWorkerPool::new(if separate_pools { 1 } else { 2 }, options)?;
    let second_pool = if separate_pools {
        UringWorkerPool::new(1, options)?
    } else {
        first_pool.clone()
    };
    let own_pool = first_pool.clone();
    let other_pool = second_pool.clone();
    let gate = Arc::new(Barrier::new(3));
    let first_gate = Arc::clone(&gate);
    let (started, start) = oneshot::channel();
    let first_file = file.clone();
    let first = first_pool.spawn(move || {
        let reader = first_file.session()?;
        reader.submit(&[request(0, 0, 4096)])?;
        started.send(()).expect("start receiver");
        first_gate.wait();
        drop(other_pool);
        drop(own_pool);
        Ok(())
    });
    start.await.expect("first worker started");
    let second_gate = Arc::clone(&gate);
    let second_file = file.clone();
    let (submitted, submit) = oneshot::channel();
    let second = second_pool.spawn(move || {
        let reader = second_file.session()?;
        reader.submit(&[request(0, 0, 4096)])?;
        assert!(submitted.send(Arc::clone(&reader)).is_ok());
        second_gate.wait();
        let ReadResult::Bytes(bytes) = reader.wait()?.result? else {
            panic!("expected bytes")
        };
        assert_eq!(bytes.try_into_host_sync()?.as_slice(), &[47; 4096]);
        Ok(())
    });
    let second_reader = submit.await.expect("second worker submitted");
    drop(first_pool);
    drop(second_pool);
    gate.wait();
    let mut finished = Box::pin(async move { futures::join!(first, second) });
    let results = tokio::time::timeout(Duration::from_secs(2), &mut finished).await;
    let Ok((first, second)) = results else {
        // Release the joined worker if shutdown regresses, so the test reports the deadlock.
        second_reader.cancel(ReadId(0));
        drop(finished.await);
        vortex_bail!("Pool shutdown blocked a worker needed by a shared read");
    };
    first?;
    second?;
    assert_eq!(file.stats().physical_reads, 1);
    assert_eq!(file.0.inflight.get().expect("registry").pending_count(), 0);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn pending_jobs_overlap_and_coalesce_on_one_worker(
    #[values(TestMode::Buffered, TestMode::Mapped)] mode: TestMode,
    #[values(false, true)] coalesce: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[31; 4096])?;
    let file = mode.open(temporary.path())?;
    let pool = UringWorkerPool::new(
        1,
        UringOptions {
            max_active_jobs: 2,
            coalesce,
            ..Default::default()
        },
    )?;
    let (started, start) = oneshot::channel();
    let (release, released) = oneshot::channel();
    let release = Arc::new(Mutex::new(Some(release)));
    let first_file = file.clone();
    let first = pool.spawn_async(move || async move {
        let owner = Rc::new(thread::current().id());
        started.send(()).expect("start receiver");
        released.await.expect("release sender");
        let reader = first_file.session()?;
        reader.submit(&[request(0, 0, 64)])?;
        assert!(reader.try_complete()?.is_none());
        assert_eq!(first_file.stats().physical_reads, 0);
        let ReadResult::Bytes(bytes) = poll_fn(|cx| reader.poll_complete(cx)).await?.result? else {
            panic!("expected bytes")
        };
        assert_eq!(thread::current().id(), *owner);
        assert_eq!(bytes.try_into_host_sync()?.as_ref(), &[31; 64]);
        Ok(*owner)
    });
    start.await.expect("first job starts");
    let second_file = file.clone();
    let second_release = Arc::clone(&release);
    let second = pool.spawn_async(move || async move {
        if let Some(release) = second_release.lock().take() {
            let _ = release.send(());
        }
        let reader = second_file.session()?;
        reader.submit(&[request(0, 64, 64)])?;
        assert!(reader.try_complete()?.is_none());
        assert_eq!(second_file.stats().physical_reads, 0);
        let ReadResult::Bytes(bytes) = poll_fn(|cx| reader.poll_complete(cx)).await?.result? else {
            panic!("expected bytes")
        };
        assert_eq!(bytes.try_into_host_sync()?.as_ref(), &[31; 64]);
        Ok(thread::current().id())
    });
    let mut joined = tokio::spawn(async move { futures::future::try_join(first, second).await });
    let result = tokio::time::timeout(Duration::from_secs(1), &mut joined).await;
    let Ok(result) = result else {
        // Release the first job so a serial implementation can shut down before this test fails.
        if let Some(release) = release.lock().take() {
            let _ = release.send(());
        }
        joined.await.expect("join jobs")?;
        vortex_bail!("The second job could not run while the first was waiting");
    };
    let (first_owner, second_owner) = result.expect("join jobs")?;
    assert_eq!(first_owner, second_owner);
    assert_eq!(file.stats().logical_reads, 2);
    assert_eq!(file.stats().physical_reads, if coalesce { 1 } else { 2 });
    Ok(())
}

struct DropSignal {
    owner: Rc<ThreadId>,
    dropped: Option<oneshot::Sender<ThreadId>>,
}

impl Drop for DropSignal {
    fn drop(&mut self) {
        assert_eq!(thread::current().id(), *self.owner);
        if let Some(sender) = self.dropped.take() {
            let _ = sender.send(*self.owner);
        }
    }
}

#[rstest]
#[tokio::test]
async fn cancel_pending_job_drops_its_future_on_its_worker(
    #[values(false, true)] control: bool,
) -> VortexResult<()> {
    let pool = UringWorkerPool::new(
        1,
        UringOptions {
            max_active_jobs: 2,
            max_control_jobs: usize::from(control),
            ..Default::default()
        },
    )?;
    let (started, start) = oneshot::channel();
    let (dropped, mut drop_signal) = oneshot::channel();
    let (rescue, rescued) = oneshot::channel();
    let work = move || async move {
        let _guard = DropSignal {
            owner: Rc::new(thread::current().id()),
            dropped: Some(dropped),
        };
        started.send(()).expect("start receiver");
        let _ = rescued.await;
        Ok(())
    };
    let job = if control {
        pool.spawn_control(work)
    } else {
        pool.spawn_async(work)
    };
    start.await.expect("job starts");
    drop(job);
    let result = tokio::time::timeout(Duration::from_secs(1), &mut drop_signal).await;
    let Ok(result) = result else {
        let _ = rescue.send(());
        drop_signal.await.expect("worker drops the rescued future");
        vortex_bail!("Dropping the handle did not cancel the pending future");
    };
    let owner = result.expect("worker drops the future");
    let next_owner = pool.spawn(|| Ok(thread::current().id())).await?;
    assert_eq!(owner, next_owner);
    Ok(())
}

#[rstest]
#[tokio::test]
async fn async_job_panics_do_not_stop_the_worker(
    #[values(false, true)] in_constructor: bool,
) -> VortexResult<()> {
    let pool = UringWorkerPool::new(
        1,
        UringOptions {
            max_active_jobs: 2,
            ..Default::default()
        },
    )?;
    let job = pool.spawn_async(move || {
        assert!(!in_constructor, "constructor panic");
        poll_fn(|_| -> Poll<VortexResult<()>> {
            panic!("poll panic");
        })
    });
    assert!(AssertUnwindSafe(job).catch_unwind().await.is_err());
    assert_eq!(pool.spawn(|| Ok(7)).await?, 7);
    Ok(())
}

#[tokio::test]
async fn pending_jobs_leave_work_for_other_workers() -> VortexResult<()> {
    let pool = UringWorkerPool::with_inherited_affinity(
        2,
        UringOptions {
            max_active_jobs: 8,
            ..Default::default()
        },
    )?;
    let mut blockers = Vec::new();
    let mut gates = Vec::new();
    for _ in 0..2 {
        let (started, start) = oneshot::channel();
        let (release, released) = crossbeam_channel::bounded(1);
        gates.push(release);
        blockers.push(pool.spawn(move || {
            let _ = started.send(());
            let _ = released.recv();
            Ok(())
        }));
        tokio::time::timeout(Duration::from_secs(1), start)
            .await
            .expect("worker starts")
            .expect("worker reports start");
    }
    let (started, mut starts) = mpsc::unbounded();
    let mut releases = Vec::new();
    let mut jobs = Vec::new();
    for _ in 0..6 {
        let (release, released) = oneshot::channel();
        releases.push(release);
        let started = started.clone();
        jobs.push(pool.spawn_async(move || async move {
            started
                .unbounded_send(thread::current().id())
                .expect("start receiver");
            let _ = released.await;
            Ok(())
        }));
    }
    gates.remove(0).send(()).expect("first worker waiting");
    let first = tokio::time::timeout(Duration::from_secs(1), starts.next())
        .await
        .expect("first job starts");
    let second = tokio::time::timeout(Duration::from_secs(1), starts.next())
        .await
        .expect("second job starts");
    assert_eq!(first, second);
    let third = tokio::time::timeout(Duration::from_millis(100), starts.next()).await;
    // Let every worker finish before reporting an admission failure.
    gates.remove(0).send(()).expect("second worker waiting");
    let remaining = if third.is_ok() { 3 } else { 4 };
    let admitted = tokio::time::timeout(Duration::from_secs(1), async {
        for _ in 0..remaining {
            starts.next().await.expect("remaining job starts");
        }
    })
    .await;
    for release in releases {
        let _ = release.send(());
    }
    futures::future::try_join_all(blockers).await?;
    futures::future::try_join_all(jobs).await?;
    assert!(
        third.is_err(),
        "One worker reserved jobs while another worker was occupied"
    );
    assert!(
        admitted.is_ok(),
        "Queued jobs stalled after admission became possible"
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn control_jobs_can_await_children_with_one_scan_slot(
    #[values(1, 2)] workers: usize,
    #[values(TestMode::Buffered, TestMode::Mapped)] mode: TestMode,
    #[values(false, true)] reserved: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[83; 4096])?;
    let file = mode.open(temporary.path())?;
    let pool = UringWorkerPool::new(
        workers + usize::from(reserved),
        UringOptions {
            max_active_jobs: 1,
            max_control_jobs: workers,
            control_workers: usize::from(reserved),
            ..Default::default()
        },
    )?;
    let roots = Arc::new(tokio::sync::Barrier::new(workers));
    let children = Arc::new(tokio::sync::Barrier::new(workers));
    let jobs = (0..workers)
        .map(|_| {
            let root_barrier = Arc::clone(&roots);
            let child_barrier = Arc::clone(&children);
            let child_pool = pool.clone();
            let file = file.clone();
            pool.spawn_control(move || async move {
                let owner = Rc::new(thread::current().id());
                root_barrier.wait().await;
                let child = child_pool
                    .spawn_async(move || async move {
                        let owner = Rc::new(thread::current().id());
                        let reader = file.session()?;
                        reader.submit(&[request(0, 0, 4096)])?;
                        let ReadResult::Bytes(bytes) =
                            poll_fn(|cx| reader.poll_complete(cx)).await?.result?
                        else {
                            panic!("expected bytes");
                        };
                        assert_eq!(bytes.try_into_host_sync()?.as_slice(), &[83; 4096]);
                        child_barrier.wait().await;
                        assert_eq!(thread::current().id(), *owner);
                        Ok(*owner)
                    })
                    .await?;
                assert_eq!(thread::current().id(), *owner);
                Ok((child, *owner))
            })
        })
        .collect::<Vec<_>>();
    let owners = tokio::time::timeout(Duration::from_secs(5), futures::future::try_join_all(jobs))
        .await
        .expect("control jobs must not consume their children's admission")?;
    if workers == 2 {
        assert_ne!(owners[0].0, owners[1].0);
    }
    if reserved {
        assert!(owners.iter().all(|(child, parent)| child != parent));
        assert!(owners.iter().all(|(_, parent)| *parent == owners[0].1));
    }
    Ok(())
}

#[rstest]
fn reserved_control_workers_require_separate_scan_capacity(
    #[values(0, 1)] max_control_jobs: usize,
) -> VortexResult<()> {
    let result = UringWorkerPool::new(
        1,
        UringOptions {
            control_workers: 1,
            max_control_jobs,
            ..Default::default()
        },
    );
    assert!(result.is_err());
    let result = UringWorkerPool::new(
        2,
        UringOptions {
            control_workers: 1,
            max_control_jobs: 0,
            ..Default::default()
        },
    );
    assert!(result.is_err());
    Ok(())
}

#[tokio::test]
async fn control_queue_does_not_raise_scan_job_limit() -> VortexResult<()> {
    let pool = UringWorkerPool::new(
        1,
        UringOptions {
            max_active_jobs: 1,
            max_control_jobs: 1,
            ..Default::default()
        },
    )?;
    let (started, start) = oneshot::channel();
    let (release, released) = oneshot::channel();
    let first = pool.spawn_async(move || async move {
        started.send(()).expect("start receiver");
        released.await.expect("release sender");
        Ok(())
    });
    start.await.expect("first leaf starts");
    let (started, mut start) = oneshot::channel();
    let second = pool.spawn_async(move || async move {
        started.send(()).expect("start receiver");
        Ok(())
    });
    let control = pool.spawn_control(move || async move {
        assert!(start.try_recv().expect("second start sender").is_none());
        release.send(()).expect("first leaf is waiting");
        start
            .await
            .expect("second leaf starts after the first completes");
        Ok(())
    });
    tokio::time::timeout(
        Duration::from_secs(5),
        futures::future::try_join3(first, second, control),
    )
    .await
    .expect("control task releases the first leaf")?;
    Ok(())
}
