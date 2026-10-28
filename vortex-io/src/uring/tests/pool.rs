// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::poll_fn;
use std::io::Write;
use std::panic::AssertUnwindSafe;
use std::rc::Rc;
use std::sync::Arc;
use std::thread;
use std::thread::ThreadId;
use std::task::Poll;
use std::time::Duration;

use futures::channel::oneshot;
use futures::channel::mpsc;
use futures::FutureExt;
use futures::StreamExt;
use parking_lot::Mutex;
use rstest::rstest;
use tempfile::NamedTempFile;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::TestMode;
use super::request;
use crate::read_at2::ReadResult;
use crate::uring::UringOptions;
use crate::uring::UringWorkerPool;

#[rstest]
#[tokio::test]
async fn pending_jobs_overlap_and_coalesce_on_one_worker(
    #[values(TestMode::Buffered, TestMode::Mapped)] mode: TestMode,
    #[values(false, true)] coalesce: bool,
) -> VortexResult<()> {
    let mut temporary = NamedTempFile::new()?;
    temporary.write_all(&[31; 4096])?;
    let file = mode.open(temporary.path())?;
    let pool = UringWorkerPool::new(1, UringOptions {
        max_active_jobs: 2, coalesce, ..Default::default()
    })?;
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
        let ReadResult::Bytes(bytes) = poll_fn(|cx| reader.poll_complete(cx)).await?.result?
            else { panic!("expected bytes") };
        assert_eq!(thread::current().id(), *owner);
        assert_eq!(bytes.try_into_host_sync()?.as_ref(), &[31; 64]);
        Ok(*owner)
    });
    start.await.expect("first job starts");
    let second_file = file.clone();
    let second_release = Arc::clone(&release);
    let second = pool.spawn_async(move || async move {
        if let Some(release) = second_release.lock().take() { let _ = release.send(()); }
        let reader = second_file.session()?;
        reader.submit(&[request(0, 64, 64)])?;
        assert!(reader.try_complete()?.is_none());
        assert_eq!(second_file.stats().physical_reads, 0);
        let ReadResult::Bytes(bytes) = poll_fn(|cx| reader.poll_complete(cx)).await?.result?
            else { panic!("expected bytes") };
        assert_eq!(bytes.try_into_host_sync()?.as_ref(), &[31; 64]);
        Ok(thread::current().id())
    });
    let mut joined = tokio::spawn(async move { futures::future::try_join(first, second).await });
    let result = tokio::time::timeout(Duration::from_secs(1), &mut joined).await;
    let Ok(result) = result else {
        // Release the first job so a serial implementation can shut down before this test fails.
        if let Some(release) = release.lock().take() { let _ = release.send(()); }
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
        if let Some(sender) = self.dropped.take() { let _ = sender.send(*self.owner); }
    }
}

#[tokio::test]
async fn cancel_pending_job_drops_its_future_on_its_worker() -> VortexResult<()> {
    let pool = UringWorkerPool::new(1, UringOptions { max_active_jobs: 2, ..Default::default() })?;
    let (started, start) = oneshot::channel();
    let (dropped, mut drop_signal) = oneshot::channel();
    let (rescue, rescued) = oneshot::channel();
    let job = pool.spawn_async(move || async move {
        let _guard = DropSignal {
            owner: Rc::new(thread::current().id()), dropped: Some(dropped),
        };
        started.send(()).expect("start receiver");
        let _ = rescued.await;
        Ok(())
    });
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
async fn async_job_panics_do_not_stop_the_worker(#[values(false, true)] in_constructor: bool) -> VortexResult<()> {
    let pool = UringWorkerPool::new(1, UringOptions { max_active_jobs: 2, ..Default::default() })?;
    let job = pool.spawn_async(move || {
        assert!(!in_constructor, "constructor panic");
        poll_fn(|_| -> Poll<VortexResult<()>> { panic!("poll panic"); })
    });
    assert!(AssertUnwindSafe(job).catch_unwind().await.is_err());
    assert_eq!(pool.spawn(|| Ok(7)).await?, 7);
    Ok(())
}

#[tokio::test]
async fn pending_jobs_leave_work_for_other_workers() -> VortexResult<()> {
    let pool = UringWorkerPool::with_inherited_affinity(2, UringOptions {
        max_active_jobs: 8, ..Default::default()
    })?;
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
        tokio::time::timeout(Duration::from_secs(1), start).await.expect("worker starts")
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
            started.unbounded_send(thread::current().id()).expect("start receiver");
            let _ = released.await;
            Ok(())
        }));
    }
    gates.remove(0).send(()).expect("first worker waiting");
    let first = tokio::time::timeout(Duration::from_secs(1), starts.next()).await.expect("first job starts");
    let second = tokio::time::timeout(Duration::from_secs(1), starts.next()).await.expect("second job starts");
    assert_eq!(first, second);
    let third = tokio::time::timeout(Duration::from_millis(100), starts.next()).await;
    // Let every worker finish before reporting an admission failure.
    gates.remove(0).send(()).expect("second worker waiting");
    let remaining = if third.is_ok() { 3 } else { 4 };
    let admitted = tokio::time::timeout(Duration::from_secs(1), async {
        for _ in 0..remaining { starts.next().await.expect("remaining job starts"); }
    }).await;
    for release in releases { let _ = release.send(()); }
    futures::future::try_join_all(blockers).await?;
    futures::future::try_join_all(jobs).await?;
    assert!(third.is_err(), "One worker reserved jobs while another worker was occupied");
    assert!(admitted.is_ok(), "Queued jobs stalled after admission became possible");
    Ok(())
}
