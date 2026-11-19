// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`DataSource`] that scans several child [`DataSource`]s as one.
//!
//! The children may be of any format: each one plans, prunes and executes its own scan, and the
//! multi source concatenates their partitions. A child is either opened up front or deferred to a
//! [`DataSourceFactory`] that opens it during the scan, so a source over many files reads only
//! the first footer before it is scanned.
//!
//! # Partitions
//!
//! Each partition reports the ordinal of the child it came from as its
//! [`index`](Partition::index), so indices are stable across scans of the same source. A
//! request's `partition_range` and `partition_selection` select children by that ordinal before
//! any deferred child is opened, so an unselected child is never opened. A child whose scan
//! yields several partitions gives all of them its ordinal.
//!
//! # Limits
//!
//! An unordered scan shares one row budget among all of its partitions, so together they return
//! at most `limit` rows, and the scan stops opening children once the budget is spent. An ordered
//! scan passes the limit to every child as a local cap, and the caller trims the concatenated
//! result, as [`ScanRequest::limit`] describes.
//!
//! # Schema
//!
//! All children must have the source's [`DType`]. A child with another dtype is an error item in
//! the partition stream.

use std::any::Any;
use std::future;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::FutureExt;
use futures::StreamExt;
use futures::stream;
use itertools::Itertools;
use tracing::Instrument;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::stats::Precision;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_io::runtime::Handle;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::DataSource;
use crate::DataSourceRef;
use crate::DataSourceScan;
use crate::DataSourceScanRef;
use crate::Partition;
use crate::PartitionRef;
use crate::PartitionStream;
use crate::ScanRequest;
use crate::selection::Selection;

/// Concurrency for opening deferred children when the machine's parallelism is unknown.
const DEFAULT_CONCURRENCY: usize = 8;

/// Opens a child of a [`MultiDataSource`] during a scan.
///
/// Implementations open the underlying data, such as a file and its footer. They return `None`
/// when the child should be skipped altogether.
#[async_trait]
pub trait DataSourceFactory: 'static + Send + Sync {
    /// Opens the child, or returns `None` if it should be skipped.
    async fn open(&self) -> VortexResult<Option<DataSourceRef>>;
}

/// A child of a [`MultiDataSource`].
#[derive(Clone)]
pub enum MultiDataSourceChild {
    /// A child that is already open.
    Opened {
        /// The child's data source.
        source: DataSourceRef,
        /// The child's on-storage size in bytes, if known.
        byte_size: Option<u64>,
    },
    /// A child opened by its factory when a scan reaches it.
    Deferred {
        /// Opens the child's data source.
        factory: Arc<dyn DataSourceFactory>,
        /// The child's on-storage size in bytes, if known.
        byte_size: Option<u64>,
    },
}

impl MultiDataSourceChild {
    /// The child's on-storage size in bytes, if known.
    pub fn byte_size(&self) -> Option<u64> {
        match self {
            Self::Opened { byte_size, .. } | Self::Deferred { byte_size, .. } => *byte_size,
        }
    }
}

/// A [`DataSource`] that scans its child [`DataSource`]s in order, as one source.
///
/// See the [module documentation](self) for how partitions, limits and dtypes behave.
#[derive(Clone)]
pub struct MultiDataSource {
    dtype: DType,
    session: VortexSession,
    children: Arc<[MultiDataSourceChild]>,
    concurrency: usize,
}

impl MultiDataSource {
    /// A source whose first child is open and whose remaining children are opened by their
    /// factories during scans.
    ///
    /// The first child determines the dtype. `byte_sizes` holds the on-storage size of each
    /// child, first then remaining, or `None` where unknown. It must be empty or have
    /// `1 + remaining.len()` entries.
    pub fn new_with_first(
        first: DataSourceRef,
        remaining: Vec<Arc<dyn DataSourceFactory>>,
        byte_sizes: Vec<Option<u64>>,
        session: &VortexSession,
    ) -> Self {
        let dtype = first.dtype().clone();
        let mut sizes = sizes_or_unknown(byte_sizes, 1 + remaining.len()).into_iter();
        let first = MultiDataSourceChild::Opened {
            source: first,
            byte_size: sizes.next().flatten(),
        };
        let children = std::iter::once(first)
            .chain(
                remaining
                    .into_iter()
                    .zip_eq(sizes)
                    .map(|(factory, byte_size)| MultiDataSourceChild::Deferred {
                        factory,
                        byte_size,
                    }),
            )
            .collect();
        Self::new(dtype, children, session)
    }

    /// A source whose children are all opened by their factories during scans.
    ///
    /// The dtype is given, so no child is opened before a scan reaches it. `byte_sizes` holds the
    /// on-storage size of each child, or `None` where unknown. It must be empty or have
    /// `factories.len()` entries.
    pub fn new_deferred(
        dtype: DType,
        factories: Vec<Arc<dyn DataSourceFactory>>,
        byte_sizes: Vec<Option<u64>>,
        session: &VortexSession,
    ) -> Self {
        let sizes = sizes_or_unknown(byte_sizes, factories.len());
        let children = factories
            .into_iter()
            .zip_eq(sizes)
            .map(|(factory, byte_size)| MultiDataSourceChild::Deferred { factory, byte_size })
            .collect();
        Self::new(dtype, children, session)
    }

    fn new(dtype: DType, children: Arc<[MultiDataSourceChild]>, session: &VortexSession) -> Self {
        Self {
            dtype,
            session: session.clone(),
            children,
            concurrency: get_available_parallelism().unwrap_or(DEFAULT_CONCURRENCY),
        }
    }

    /// Sets how many children a scan opens and plans at once. Defaults to the number of
    /// available cores.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// The children, in partition-index order.
    pub fn children(&self) -> &[MultiDataSourceChild] {
        &self.children
    }
}

/// `byte_sizes`, or `count` unknown sizes when it is empty.
fn sizes_or_unknown(byte_sizes: Vec<Option<u64>>, count: usize) -> Vec<Option<u64>> {
    if byte_sizes.is_empty() {
        vec![None; count]
    } else {
        debug_assert_eq!(
            byte_sizes.len(),
            count,
            "byte_sizes must have one entry per child"
        );
        byte_sizes
    }
}

/// Sums `known` estimates and extrapolates their mean over all `total` children.
///
/// The sum is exact only when every child is known exactly.
fn extrapolate(known: impl Iterator<Item = Precision<u64>>, total: usize) -> Precision<u64> {
    if total == 0 {
        return Precision::exact(0u64);
    }
    let (mut sum, mut count, mut exact) = (0u64, 0u64, true);
    for estimate in known {
        match estimate {
            Precision::Exact(value) => sum = sum.saturating_add(value),
            Precision::Inexact(value) => {
                sum = sum.saturating_add(value);
                exact = false;
            }
            Precision::Absent => continue,
        }
        count += 1;
    }
    let total = total as u64;
    if count == 0 {
        Precision::Absent
    } else if count == total && exact {
        Precision::exact(sum)
    } else if count == total {
        Precision::inexact(sum)
    } else {
        Precision::inexact((sum / count).saturating_mul(total))
    }
}

#[async_trait]
impl DataSource for MultiDataSource {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// The sum of the opened children's row counts, extrapolated over the deferred children.
    fn row_count(&self) -> Precision<u64> {
        let opened = self.children.iter().map(|child| match child {
            MultiDataSourceChild::Opened { source, .. } => source.row_count(),
            MultiDataSourceChild::Deferred { .. } => Precision::Absent,
        });
        extrapolate(opened, self.children.len())
    }

    /// The sum of the known child sizes, extrapolated over the unknown ones.
    fn byte_size(&self) -> Precision<u64> {
        let known = self.children.iter().map(|child| {
            child
                .byte_size()
                .map_or(Precision::Absent, Precision::exact)
        });
        extrapolate(known, self.children.len())
    }

    fn deserialize_partition(
        &self,
        _data: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        vortex_bail!("MultiDataSource partitions are not yet serializable")
    }

    async fn scan(&self, request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        // Only the projection is bound here, for the scan's dtype. Each child binds the request
        // itself, so a filter that does not bind fails the child's scan, not this one.
        let dtype = request.projection.bind(&self.dtype)?.dtype().clone();

        let children = self
            .children
            .iter()
            .enumerate()
            .filter(|(ordinal, _)| selects(&request, *ordinal as u64))
            .map(|(ordinal, child)| (ordinal, child.clone()))
            .collect();

        // Only an unordered scan shares its limit: an ordered one cannot take rows in completion
        // order, so each child applies the limit locally and the caller trims the result.
        let budget = request
            .limit
            .filter(|_| !request.ordered)
            .map(RowBudget::new);

        let ScanRequest {
            projection,
            filter,
            row_range,
            selection,
            partition_selection: _,
            partition_range: _,
            ordered,
            limit,
            split_concurrency,
        } = request;
        // Partitions are selected by child above, so every selected child scans in full.
        let request = ScanRequest {
            projection,
            filter,
            row_range,
            selection,
            partition_selection: Selection::default(),
            partition_range: None,
            ordered,
            limit,
            split_concurrency,
        };

        Ok(Box::new(MultiScan {
            dtype,
            source_dtype: self.dtype.clone(),
            request,
            budget,
            children,
            handle: self.session.handle(),
            concurrency: self.concurrency,
        }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

/// Whether `request` selects the child with this ordinal.
fn selects(request: &ScanRequest, ordinal: u64) -> bool {
    request
        .partition_range
        .as_ref()
        .is_none_or(|range| range.contains(&ordinal))
        && request
            .partition_selection
            .row_count_in_range(&(ordinal..ordinal + 1))
            == 1
}

/// A row budget shared by all partitions of an unordered scan.
#[derive(Clone)]
struct RowBudget(Arc<AtomicU64>);

impl RowBudget {
    fn new(limit: u64) -> Self {
        Self(Arc::new(AtomicU64::new(limit)))
    }

    /// Reserves up to `rows` rows, returning how many the remaining budget granted.
    fn reserve(&self, rows: u64) -> u64 {
        let remaining = self
            .0
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                Some(remaining.saturating_sub(rows))
            })
            .unwrap_or_else(|remaining| remaining);
        remaining.min(rows)
    }

    fn is_exhausted(&self) -> bool {
        self.0.load(Ordering::Relaxed) == 0
    }
}

fn is_exhausted(budget: Option<&RowBudget>) -> bool {
    budget.is_some_and(RowBudget::is_exhausted)
}

struct MultiScan {
    dtype: DType,
    source_dtype: DType,
    request: ScanRequest,
    budget: Option<RowBudget>,
    children: Vec<(usize, MultiDataSourceChild)>,
    handle: Handle,
    concurrency: usize,
}

impl DataSourceScan for MultiScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// One partition per selected child, but a child may prune itself to none or yield several.
    fn partition_count(&self) -> Precision<usize> {
        if self.children.is_empty() {
            Precision::exact(0usize)
        } else {
            Precision::inexact(self.children.len())
        }
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        let Self {
            dtype: _,
            source_dtype,
            request,
            budget,
            children,
            handle,
            concurrency,
        } = *self;

        let ordered = request.ordered;
        let open_budget = budget.clone();
        let scans = stream::iter(children)
            // Stop opening children once an unordered scan's budget is spent.
            .take_while(move |_| future::ready(!is_exhausted(open_budget.as_ref())))
            .map(move |(ordinal, child)| {
                let scan = scan_child(ordinal, child, source_dtype.clone(), request.clone());
                handle.spawn(scan).map(move |scan| (ordinal, scan))
            });
        // An ordered scan keeps the children in order; an unordered one takes whichever is
        // planned first.
        let scans = if ordered {
            scans.buffered(concurrency).boxed()
        } else {
            scans.buffer_unordered(concurrency).boxed()
        };

        let partition_budget = budget.clone();
        scans
            .flat_map(move |(ordinal, scan)| match scan {
                Ok(Some(scan)) => {
                    let budget = budget.clone();
                    scan.partitions()
                        .map(move |partition| {
                            partition.map(|inner| {
                                Box::new(MultiPartition {
                                    inner,
                                    index: ordinal,
                                    budget: budget.clone(),
                                }) as PartitionRef
                            })
                        })
                        .boxed()
                }
                Ok(None) => stream::empty().boxed(),
                Err(error) => stream::once(future::ready(Err(error))).boxed(),
            })
            // A partition planned after the budget is spent has no rows left to return.
            .take_while(move |partition| {
                future::ready(partition.is_err() || !is_exhausted(partition_budget.as_ref()))
            })
            .boxed()
    }
}

/// Opens the child with this ordinal if it is deferred, checks its dtype and plans its scan.
///
/// Returns `None` if the child's factory skips it.
async fn scan_child(
    ordinal: usize,
    child: MultiDataSourceChild,
    dtype: DType,
    request: ScanRequest,
) -> VortexResult<Option<DataSourceScanRef>> {
    let source = match child {
        MultiDataSourceChild::Opened { source, .. } => source,
        MultiDataSourceChild::Deferred { factory, .. } => {
            let opened = factory
                .open()
                .instrument(tracing::info_span!("DataSourceFactory::open", ordinal))
                .await?;
            let Some(source) = opened else {
                return Ok(None);
            };
            source
        }
    };
    vortex_ensure_eq!(
        source.dtype(),
        &dtype,
        "child {ordinal} of a multi data source has another dtype"
    );
    source.scan(request).await.map(Some)
}

/// A partition of a child, reporting the child's ordinal as its index and taking its rows from
/// the scan's shared budget, if any.
struct MultiPartition {
    inner: PartitionRef,
    index: usize,
    budget: Option<RowBudget>,
}

impl Partition for MultiPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index(&self) -> usize {
        self.index
    }

    fn row_count(&self) -> Precision<u64> {
        let row_count = self.inner.row_count();
        if self.budget.is_some() {
            row_count.into_inexact()
        } else {
            row_count
        }
    }

    fn byte_size(&self) -> Precision<u64> {
        self.inner.byte_size()
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let arrays = self.inner.execute()?;
        let Some(budget) = self.budget else {
            return Ok(arrays);
        };
        let dtype = arrays.dtype().clone();
        // Each array takes its rows from the budget, and the stream ends once it is spent.
        let arrays = arrays.scan(false, move |done, array| {
            if *done || budget.is_exhausted() {
                return future::ready(None);
            }
            let array = match array {
                Ok(array) => array,
                Err(error) => return future::ready(Some(Err(error))),
            };
            let rows = array.len() as u64;
            let granted = budget.reserve(rows);
            if granted == rows {
                return future::ready(Some(Ok(array)));
            }
            *done = true;
            if granted == 0 {
                return future::ready(None);
            }
            let granted = usize::try_from(granted).unwrap_or(usize::MAX);
            future::ready(Some(array.slice(0..granted)))
        });
        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype, arrays,
        )))
    }
}

#[cfg(test)]
mod tests;
