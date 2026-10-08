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
                                        let work = async move {
                                            if job.result.is_canceled() {
                                                return;
                                            }
                                            let started = Instant::now();
                                            tracing::debug!(
                                                file_index = job.index,
                                                "file preparation started"
                                            );
                                            let result = async {
                                                let mut file = OpenFileReader::open(job.path).await?;
                                                prepare_reader(&mut file, &projection, &filter).await?;
                                                Ok(file)
                                            }
                                            .await;
                                            tracing::debug!(
                                                file_index = job.index,
                                                elapsed_us = started.elapsed().as_micros(),
                                                "file preparation finished"
                                            );
                                            drop(job.result.send(result));
                                        };
                                        futures::pin_mut!(work);
                                        drop(futures::future::select(cancelled, work).await);
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
        let file = RUNTIME.block_on(result)
            .map_err(|_| vortex_err!("file preparation driver stopped"))??;
        tracing::debug!(
            file_index = index,
            wait_us = started.elapsed().as_micros(),
            "prepared file claimed"
        );
        Ok(Some(file))
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
