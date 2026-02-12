// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`DataSource`] that combines multiple child data sources into a single scannable source.
//!
//! Splits from all children are interleaved, enabling parallel execution across files.
//! Children can be pre-opened (eager) or opened lazily via [`DataSourceFactory`] implementations,
//! with spawned prefetching to overlap file-opening I/O with split execution.
//!
//! # Future Work
//!
//! This data source should evolve to support hive-style partitioning columns, different strategies
//! for unifying schemas, more flexible prefetching configurations, and more robust error handling
//! (e.g., skip failed sources instead of aborting the entire scan).

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream;
use parking_lot::Mutex;
use tracing::Instrument;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::api::DataSource;
use crate::api::DataSourceRef;
use crate::api::DataSourceScan;
use crate::api::DataSourceScanRef;
use crate::api::Estimate;
use crate::api::ScanRequest;
use crate::api::SplitRef;
use crate::api::SplitStream;

/// An async factory that produces a [`DataSource`].
///
/// Implementations handle engine-specific concerns like file opening, caching, and
/// statistics-based pruning. Returns `None` if the source should be skipped (e.g., pruned
/// based on file-level statistics).
#[async_trait]
pub trait DataSourceFactory: 'static + Send + Sync {
    /// Opens the data source, or returns `None` if it should be skipped.
    async fn open(&self) -> VortexResult<Option<DataSourceRef>>;
}

/// Default number of deferred sources to open concurrently during scanning.
const DEFAULT_PREFETCH: usize = 8;

/// A [`DataSource`] combining multiple children into a single scannable source.
///
/// Children may be pre-opened or deferred via [`DataSourceFactory`]. During scanning,
/// deferred children are opened in the background using spawned tasks on the session's runtime,
/// keeping the I/O pipeline full while the engine processes splits from already-open sources.
///
/// Once a deferred child is successfully opened, it is stored so that subsequent scans reuse the
/// opened source without re-opening.
pub struct MultiDataSource {
    dtype: DType,
    children: Arc<Mutex<Vec<MultiChild>>>,
    handle: Handle,
    prefetch: usize,
}

enum MultiChild {
    Opened(DataSourceRef),
    Deferred(Arc<dyn DataSourceFactory>),
}

impl MultiDataSource {
    /// Creates a multi-source from pre-opened data sources.
    ///
    /// Validates that all children share the same dtype.
    pub fn try_new(children: Vec<DataSourceRef>, session: &VortexSession) -> VortexResult<Self> {
        let first = children
            .first()
            .ok_or_else(|| vortex_err!("MultiDataSource requires at least one child"))?;
        let dtype = first.dtype().clone();

        for (i, child) in children.iter().enumerate().skip(1) {
            if child.dtype() != &dtype {
                vortex_bail!(
                    "MultiDataSource dtype mismatch in child {}: expected {}, got {}",
                    i,
                    dtype,
                    child.dtype()
                );
            }
        }

        Ok(Self {
            dtype,
            children: Arc::new(Mutex::new(
                children.into_iter().map(MultiChild::Opened).collect(),
            )),
            handle: session.handle(),
            prefetch: std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(DEFAULT_PREFETCH),
        })
    }

    /// Creates a multi-source with lazy opening.
    ///
    /// The first source must be pre-opened to determine the dtype (required by the sync
    /// [`DataSource::dtype`] method). Remaining sources are opened lazily during scanning
    /// via their factories, with dtype validated on open.
    pub fn lazy(
        first: DataSourceRef,
        remaining: Vec<Arc<dyn DataSourceFactory>>,
        session: &VortexSession,
    ) -> Self {
        let dtype = first.dtype().clone();
        let mut children = Vec::with_capacity(1 + remaining.len());
        children.push(MultiChild::Opened(first));
        children.extend(remaining.into_iter().map(MultiChild::Deferred));

        Self {
            dtype,
            children: Arc::new(Mutex::new(children)),
            handle: session.handle(),
            prefetch: DEFAULT_PREFETCH,
        }
    }

    /// Sets the number of deferred sources to open concurrently during scanning.
    ///
    /// Higher values overlap more file-opening I/O with split execution but use more memory
    /// for in-flight metadata. Defaults to 8.
    pub fn with_prefetch(mut self, prefetch: usize) -> Self {
        self.prefetch = prefetch;
        self
    }
}

impl DataSource for MultiDataSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count_estimate(&self) -> Estimate<u64> {
        let mut lower: u64 = 0;
        let mut upper: Option<u64> = Some(0);
        let mut has_deferred = false;

        let children = self.children.lock();
        for child in children.iter() {
            match child {
                MultiChild::Opened(ds) => {
                    let est = ds.row_count_estimate();
                    lower = lower.saturating_add(est.lower);
                    upper = match (upper, est.upper) {
                        (Some(a), Some(b)) => Some(a.saturating_add(b)),
                        _ => None,
                    };
                }
                MultiChild::Deferred(_) => {
                    has_deferred = true;
                }
            }
        }

        if has_deferred {
            upper = None;
        }

        Estimate { lower, upper }
    }

    fn deserialize_split(&self, _data: &[u8], _session: &VortexSession) -> VortexResult<SplitRef> {
        vortex_bail!("MultiDataSource splits are not yet serializable")
    }

    fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let mut ready = VecDeque::new();
        let mut deferred = VecDeque::new();

        let children = self.children.lock();
        for (i, child) in children.iter().enumerate() {
            match child {
                MultiChild::Opened(ds) => ready.push_back(ds.clone()),
                MultiChild::Deferred(factory) => deferred.push_back((i, factory.clone())),
            }
        }
        drop(children);

        let remaining_limit = scan_request.limit;

        let mut scan = MultiDataSourceScan {
            dtype: self.dtype.clone(),
            request: scan_request,
            current: None,
            ready,
            opening: VecDeque::new(),
            deferred,
            children: Arc::clone(&self.children),
            handle: self.handle.clone(),
            prefetch: self.prefetch,
            remaining_limit,
        };

        // Kick off initial prefetch of deferred sources.
        scan.fill_pipeline();

        Ok(Box::new(scan))
    }
}

struct MultiDataSourceScan {
    dtype: DType,
    request: ScanRequest,
    /// Currently active child scan being drained.
    current: Option<DataSourceScanRef>,
    /// Pre-opened sources ready to be scanned.
    ready: VecDeque<DataSourceRef>,
    /// In-flight spawned opens. Each task yields `(child_index, source)` on success.
    opening: VecDeque<Task<VortexResult<Option<(usize, DataSourceRef)>>>>,
    /// Remaining factories not yet spawned, paired with their child index.
    deferred: VecDeque<(usize, Arc<dyn DataSourceFactory>)>,
    /// Shared children vec for promoting Deferred → Opened.
    children: Arc<Mutex<Vec<MultiChild>>>,
    /// Runtime handle for spawning prefetch tasks.
    handle: Handle,
    /// Target number of in-flight + ready sources.
    prefetch: usize,
    /// Remaining row limit across all children. Decremented conservatively by each split's
    /// upper row estimate. The engine enforces the exact limit at the stream level.
    remaining_limit: Option<u64>,
}

impl MultiDataSourceScan {
    /// Spawns open tasks for deferred factories up to the prefetch target.
    fn fill_pipeline(&mut self) {
        while self.opening.len() + self.ready.len() < self.prefetch {
            let Some((idx, factory)) = self.deferred.pop_front() else {
                break;
            };
            self.opening.push_back(self.handle.spawn(async move {
                let source = factory
                    .open()
                    .instrument(tracing::info_span!("DataSourceFactory::open"))
                    .await?;
                Ok(source.map(|s| (idx, s)))
            }));
        }
    }

    /// Gets the next ready data source, awaiting in-flight opens if needed.
    async fn next_source(&mut self) -> VortexResult<Option<DataSourceRef>> {
        loop {
            if let Some(source) = self.ready.pop_front() {
                return Ok(Some(source));
            }

            if let Some(task) = self.opening.pop_front() {
                self.fill_pipeline();
                match task.await? {
                    Some((idx, source)) => {
                        // Promote Deferred → Opened so future scans reuse this source.
                        self.children.lock()[idx] = MultiChild::Opened(source.clone());
                        return Ok(Some(source));
                    }
                    None => continue, // pruned, try next
                }
            }

            return Ok(None);
        }
    }
}

impl DataSourceScan for MultiDataSourceScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn split_count_estimate(&self) -> Estimate<usize> {
        let current_estimate = self
            .current
            .as_ref()
            .map_or_else(|| Estimate::exact(0), |s| s.split_count_estimate());

        let remaining_sources = self.ready.len() + self.opening.len() + self.deferred.len();
        if remaining_sources == 0 {
            return current_estimate;
        }

        // With remaining sources whose split counts are unknown, we can only provide a lower bound.
        Estimate {
            lower: current_estimate.lower,
            upper: None,
        }
    }

    fn splits(self: Box<Self>) -> SplitStream {
        stream::unfold(
            (Some(*self), None::<SplitStream>),
            |(mut state, mut current_stream)| async move {
                loop {
                    // Try to pull from the current child's split stream.
                    if let Some(ref mut child_stream) = current_stream {
                        match child_stream.next().await {
                            Some(Ok(split)) => {
                                if let Some(ref mut s) = state
                                    && let Some(ref mut limit) = s.remaining_limit
                                {
                                    let est = split.row_count_estimate();
                                    *limit = limit.saturating_sub(est.upper.unwrap_or(est.lower));
                                }
                                return Some((Ok(split), (state, current_stream)));
                            }
                            Some(Err(e)) => {
                                return Some((Err(e), (None, None)));
                            }
                            None => {
                                // Current child exhausted, move to next.
                                drop(current_stream.take());
                            }
                        }
                    }

                    let s = state.as_mut()?;

                    if s.remaining_limit.is_some_and(|l| l == 0) {
                        return None;
                    }

                    // Get the next data source.
                    let source = match s.next_source().await {
                        Ok(Some(source)) => source,
                        Ok(None) => return None,
                        Err(e) => return Some((Err(e), (None, None))),
                    };

                    if source.dtype() != &s.dtype {
                        return Some((
                            Err(vortex_err!(
                                "MultiDataSource dtype mismatch: expected {}, got {}",
                                s.dtype,
                                source.dtype()
                            )),
                            (None, None),
                        ));
                    }

                    let mut child_request = s.request.clone();
                    child_request.limit = s.remaining_limit;
                    let child_scan = match source.scan(child_request) {
                        Ok(scan) => scan,
                        Err(e) => return Some((Err(e), (None, None))),
                    };

                    current_stream = Some(child_scan.splits());
                }
            },
        )
        .boxed()
    }
}
