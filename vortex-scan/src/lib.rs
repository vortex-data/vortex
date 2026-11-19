// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! The Vortex Scan API implements an abstract table scan interface that can be used to
//! read data from various data sources.
//!
//! It supports arbitrary projection expressions, filter expressions, and limit pushdown.
//!
//! A [`DataSource`] plans a [`Scan`]. The host decides parallelism: it may [`Scan::split`] the
//! scan into independent pieces (for example to ship them to other machines, or to keep row
//! order across threads) and [`Scan::open`] each piece into a [`ScanPool`]. Any number of
//! workers, created with [`ScanPool::worker`], then share the pool's work. A worker is a stream
//! polled inline by its caller: Vortex never spawns scan work onto threads the host did not give
//! it. [`execute_with_workers`] is the one exception, for callers that want Vortex to provide the
//! threads.
//!
//! The API is currently under development and may change in future releases, however we hope to
//! stabilize into stable C ABI for use within foreign language bindings.
//!
//! If you are looking to scan Vortex files or layouts, the Vortex implementation of the Scan API
//! is lowered by `vortex-layout` into this crate's plans and executor.
//!
//! ## Open Issues
//!
//! * We probably want to make the DataSource serializable as well, so that we can share
//!   source-level state with workers, separately from scan serialization.
//! * We should add a way for the client to negotiate capabilities with the data source, for
//!   example which encodings it knows about.

pub mod arrays;
pub mod execution;
pub mod plan;
pub mod planning;
pub mod prepare;
pub mod row_mask;
pub mod selection;
pub mod source;
pub mod strict_sorted_buffer;
mod workers;

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use selection::Selection;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::stats::StatsSet;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
pub use workers::execute_with_workers;

/// Opens a Vortex [`DataSource`] from a URI.
///
/// Configuration can be passed via the URI query parameters, similar to JDBC / ADBC.
/// Providers can be registered with the [`VortexSession`] to support additional URI schemes.
#[async_trait]
pub trait DataSourceOpener: 'static {
    /// Attempt to open a new data source from a URI.
    async fn open(&self, uri: String, session: &VortexSession) -> VortexResult<DataSourceRef>;
}

/// Supports deserialization of a Vortex [`DataSource`] on a remote worker.
#[async_trait]
pub trait DataSourceRemote: 'static {
    /// Attempt to deserialize the source.
    fn deserialize_data_source(
        &self,
        data: &[u8],
        session: &VortexSession,
    ) -> VortexResult<DataSourceRef>;
}

/// A reference-counted data source.
pub type DataSourceRef = Arc<dyn DataSource>;

/// A data source represents a dataset that can be scanned with projection and filter
/// expressions.
///
/// The source decides the units of a scan, such as layout splits or files, and prunes the units
/// that statistics exclude. It may be used many times to plan many scans. Each [`Scan`] is
/// consumed by splitting or opening it.
#[async_trait]
pub trait DataSource: 'static + Send + Sync {
    /// Returns the dtype of the source.
    fn dtype(&self) -> &DType;

    /// Returns an estimate of the row count of the un-filtered source.
    fn row_count(&self) -> Precision<u64> {
        Precision::Absent
    }

    /// Returns an estimate of the byte size of the un-filtered source.
    fn byte_size(&self) -> Precision<u64> {
        Precision::Absent
    }

    /// Serialize the [`DataSource`] to pass to a remote worker.
    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Plans a scan of the source.
    ///
    /// The request's expressions are unbound; the source binds them against [`Self::dtype`] once,
    /// here. Planning reads metadata only.
    async fn scan(&self, request: ScanRequest) -> VortexResult<ScanRef>;

    /// Deserializes a scan previously serialized by [`Scan::serialize`] from a compatible source.
    fn deserialize_scan(&self, data: &[u8], session: &VortexSession) -> VortexResult<ScanRef> {
        let _ = (data, session);
        vortex_bail!("DataSource does not support scan deserialization")
    }

    /// Returns the statistics for a given field.
    async fn field_statistics(&self, field_path: &FieldPath) -> VortexResult<StatsSet>;

    /// Returns a stored aggregate partial over the unfiltered source field.
    ///
    /// The scalar has the aggregate's `partial_dtype`; callers merge partials and finalize
    /// through the requested aggregate. Identity includes bound options. Missing partials,
    /// incomplete coverage, and incompatible aggregates return [`Precision::Absent`], as
    /// does the default implementation. Finalized legacy statistics are not partials.
    ///
    /// Lookup must not scan payloads. Precision describes the partial state; consumers must
    /// respect the aggregate's approximation contract. Later scan filters and limits do not
    /// change this source-level partial. A stored null partial is distinct from absence.
    async fn field_statistic(
        &self,
        _field_path: &FieldPath,
        _aggregate_fn: &AggregateFnRef,
    ) -> VortexResult<Precision<Scalar>> {
        Ok(Precision::Absent)
    }
}

/// A request to scan a data source: what to read.
///
/// How the scan runs (how many workers, how much work each has in flight) is not part of the
/// request: the host decides the workers and the source decides the work in flight.
#[derive(Debug, Clone)]
pub struct ScanRequest {
    /// Projection expression, unbound. Defaults to `root()` which returns all columns.
    pub projection: Expression,
    /// Filter expression, unbound. `None` implies no filter.
    pub filter: Option<Expression>,
    /// The row range to read. `None` reads every row.
    pub row_range: Option<Range<u64>>,
    /// The rows to read, by absolute row index, intersected with `row_range`.
    pub selection: Selection,
    /// Whether a single worker must yield rows in row order. If false, the scan may produce rows
    /// in any order.
    pub ordered: bool,
    /// Optional limit on the number of rows returned, applied after filtering and row selection.
    ///
    /// The limit is global: every worker of the scan's pool draws from one budget, so together
    /// they return no more than `limit` rows. An ordered scan read by one worker returns exactly
    /// its first `limit` rows.
    pub limit: Option<u64>,
}

impl Default for ScanRequest {
    fn default() -> Self {
        Self {
            projection: root(),
            filter: None,
            row_range: None,
            selection: Selection::default(),
            ordered: false,
            limit: None,
        }
    }
}

/// A boxed, planned scan.
pub type ScanRef = Box<dyn Scan>;

/// A planned scan. Split it into independent pieces, or open it to run.
pub trait Scan: 'static + Send {
    /// The dtype of the arrays the scan returns.
    fn dtype(&self) -> &DType;

    /// Whether a single worker of this scan yields rows in row order.
    fn ordered(&self) -> bool;

    /// Estimates the rows this scan returns, from plan-time state only.
    fn row_count(&self) -> Precision<u64>;

    /// Estimates the bytes this scan reads, before any filter, from plan-time state only.
    fn byte_size(&self) -> Precision<u64>;

    /// Cuts the scan into at most `n` independent scans, in row order, on unit boundaries, from
    /// metadata only.
    ///
    /// Together the pieces read exactly what this scan reads. Returns fewer pieces when the scan
    /// has fewer units, and the scan itself when it cannot be cut (for example a scan with a
    /// limit). Never returns an empty list.
    fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<ScanRef>>;

    /// Serializes this scan for a remote worker. `Ok(None)` when unsupported (the default).
    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    /// Opens the scan for execution.
    ///
    /// Must be cheap: all I/O happens when the pool's workers are polled.
    fn open(self: Box<Self>) -> VortexResult<ScanPoolRef>;
}

/// A reference-counted, opened scan.
pub type ScanPoolRef = Arc<dyn ScanPool>;

/// An opened scan whose work any number of workers share.
///
/// Each unit of work is read by exactly one worker. All workers of a pool share its row limit.
pub trait ScanPool: 'static + Send + Sync {
    /// The dtype of the arrays the workers return.
    fn dtype(&self) -> &DType;

    /// Returns a new worker: a stream of the scan's arrays that the calling thread polls inline.
    ///
    /// A worker never spawns work onto other threads. Workers may be created at any time and
    /// polled from any thread; a worker ends once the pool has no work left for it. For an
    /// ordered scan, a single worker yields rows in row order; several workers give no global
    /// order.
    fn worker(self: Arc<Self>) -> SendableArrayStream;

    /// True once every unit has been handed to a worker (workers may still be producing).
    ///
    /// A host uses it to stop adding threads to a scan that has nothing left to hand out.
    fn is_exhausted(&self) -> bool;

    /// Fraction in `[0, 1]` of the scan's units, weighted by row count, handed out so far.
    fn progress(&self) -> f64;
}

/// Expressions referring to the global scan row index.
pub mod row_idx {
    pub use crate::plan::builtins::row_idx::expr::*;
}

#[cfg(test)]
mod test;
