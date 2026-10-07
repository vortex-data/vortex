// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Query-scoped, bounded file preparation ahead of DuckDB's scan workers.

use std::collections::BTreeMap;
use std::thread::JoinHandle;
use std::time::Instant;

use futures::FutureExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::channel::oneshot;
use futures::future::Either;
use futures::future::select;
use parking_lot::Mutex;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::BoundExpression;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan;

use crate::RUNTIME;
use crate::file_reader::OpenFileReader;
use crate::file_reader::prepare_reader;
use crate::projection::Filter;

type PreparedFile = VortexResult<OpenFileReader>;

struct Job {
    index: usize,
    path: String,
    result: oneshot::Sender<PreparedFile>,
}

pub(crate) struct FilePrefetch {
    window: usize,
    jobs: mpsc::UnboundedSender<Job>,
    results: Mutex<BTreeMap<usize, oneshot::Receiver<PreparedFile>>>,
    cancel: Option<oneshot::Sender<()>>,
    driver: Option<JoinHandle<()>>,
}

impl FilePrefetch {
    pub(crate) fn new(projection: BoundExpression, filter: Filter) -> VortexResult<Option<Self>> {
        let window = std::env::var("VORTEX_DUCKDB_FILE_PREFETCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
            .min(256);
        if window == 0 || !scan::v2::enabled() {
            return Ok(None);
        }
        let (jobs, receiver) = mpsc::unbounded::<Job>();
        let (cancel, cancelled) = oneshot::channel();
        // The shared current-thread runtime otherwise stops progressing whenever DuckDB
        // is processing a result outside reader_scan's block_on.
        let driver = std::thread::Builder::new()
            .name("vortex-file-prefetch".into())
            .spawn(move || {
                RUNTIME.block_on(async move {
                    let cancelled = cancelled.shared();
                    receiver
                        .take_until(cancelled.clone())
                        .for_each_concurrent(window, |job| {
                            let projection = projection.clone();
                            let filter = filter.clone();
                            let cancelled = cancelled.clone();
                            async move {
                                // Shared executor tasks can also run on DuckDB workers inside block_on.
                                // Await each task after cancellation before joining the driver.
                                RUNTIME
                                    .handle()
                                    .spawn(async move {
                                        let Job {
                                            index,
                                            path,
                                            result,
                                        } = job;
                                        let work = async move {
                                            let started = Instant::now();
                                            tracing::debug!(
                                                file_index = index,
                                                "file preparation started"
                                            );
                                            let prepared = async {
                                                let mut file = OpenFileReader::open(path).await?;
                                                prepare_reader(&mut file, &projection, &filter)
                                                    .await?;
                                                Ok(file)
                                            }
                                            .await;
                                            tracing::debug!(
                                                file_index = index,
                                                elapsed_us = started.elapsed().as_micros(),
                                                "file preparation finished"
                                            );
                                            prepared
                                        };
                                        let work = prepare_while_needed(result, work);
                                        futures::pin_mut!(work);
                                        drop(select(cancelled, work).await);
                                    })
                                    .await;
                            }
                        })
                        .await;
                });
            })?;
        Ok(Some(Self {
            window,
            jobs,
            results: Mutex::new(BTreeMap::new()),
            cancel: Some(cancel),
            driver: Some(driver),
        }))
    }

    pub(crate) fn window(&self) -> usize {
        self.window
    }

    // C++ submits a sliding window using DuckDB's file ordinals. Completed results
    // remain in that same window, so a fast producer cannot prepare the whole dataset.
    pub(crate) fn submit(&self, index: usize, path: String) -> VortexResult<()> {
        let (result, receiver) = oneshot::channel();
        self.results.lock().insert(index, receiver);
        self.jobs
            .unbounded_send(Job {
                index,
                path,
                result,
            })
            .map_err(|_| vortex_err!("file preparation driver stopped"))
    }

    pub(crate) fn take(&self, index: usize, skip: bool) -> VortexResult<Option<OpenFileReader>> {
        let result = self
            .results
            .lock()
            .remove(&index)
            .ok_or_else(|| vortex_err!("missing prepared file {index}"))?;
        if skip {
            return Ok(None);
        }
        let started = Instant::now();
        let file = RUNTIME
            .block_on(result)
            .map_err(|_| vortex_err!("file preparation driver stopped"))??;
        tracing::debug!(
            file_index = index,
            wait_us = started.elapsed().as_micros(),
            "prepared file claimed"
        );
        Ok(Some(file))
    }
}

// Dropping a skipped file's receiver cancels both queued and in-flight preparation.
async fn prepare_while_needed<T>(mut result: oneshot::Sender<T>, work: impl Future<Output = T>) {
    futures::pin_mut!(work);
    let prepared = match select(result.cancellation(), work).await {
        Either::Left(_) => None,
        Either::Right((prepared, _)) => Some(prepared),
    };
    if let Some(prepared) = prepared {
        drop(result.send(prepared));
    }
}

impl Drop for FilePrefetch {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
        // Join before DuckDB destroys the dynamic-filter state referenced by expressions.
        if let Some(driver) = self.driver.take() {
            drop(driver.join());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;
    use std::task::Context;
    use std::task::Poll;
    use std::task::Waker;

    use futures::FutureExt;
    use futures::channel::oneshot;
    use futures::executor::block_on;
    use futures::future;
    use vortex::error::VortexResult;
    use vortex::error::vortex_err;

    use super::prepare_while_needed;

    struct Dropped(Arc<AtomicBool>);

    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }

    #[test]
    fn test_skipped_file_does_not_start_preparation() {
        let (result, receiver) = oneshot::channel::<()>();
        drop(receiver);
        block_on(prepare_while_needed(result, async {
            panic!("a skipped file must not be opened");
        }));
    }

    #[test]
    fn test_skipped_file_cancels_in_flight_preparation() {
        let (result, receiver) = oneshot::channel::<()>();
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = Dropped(Arc::clone(&dropped));
        let mut preparation = Box::pin(prepare_while_needed(result, async move {
            let _guard = guard;
            future::pending::<()>().await;
        }));
        let mut cx = Context::from_waker(Waker::noop());
        assert_eq!(preparation.poll_unpin(&mut cx), Poll::Pending);
        assert!(!dropped.load(Ordering::Relaxed));

        drop(receiver);
        assert_eq!(preparation.poll_unpin(&mut cx), Poll::Ready(()));
        assert!(dropped.load(Ordering::Relaxed));
    }

    #[test]
    fn test_preparation_error_reaches_the_scan_worker() -> VortexResult<()> {
        let (result, receiver) = oneshot::channel::<VortexResult<()>>();
        block_on(prepare_while_needed(
            result,
            future::ready(Err(vortex_err!("file preparation failed"))),
        ));
        let error = block_on(receiver)
            .map_err(|error| vortex_err!("preparation result channel closed: {error}"))?
            .expect_err("preparation should fail");
        assert!(error.to_string().contains("file preparation failed"));
        Ok(())
    }
}
