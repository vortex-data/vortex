// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for Vortex scans.
//!
//! The hand-written C ABI exposed two `box_wrapper!` opaques (`vx_scan` and `vx_partition`,
//! each a state machine over Pending/Started/Finished), the POD structs `vx_scan_options`,
//! `vx_scan_selection`, and `vx_estimate`, the C enums `vx_scan_selection_include` and
//! `vx_estimate_type`, and the free functions `vx_data_source_scan`, `vx_scan_dtype`,
//! `vx_scan_next_partition`, `vx_partition_row_count`, `vx_partition_scan_arrow`, and
//! `vx_partition_next`.
//!
//! ## Builder + async modelling
//!
//! Diplomat cannot express the C ABI's caller-populated POD `vx_scan_options` (with embedded
//! pointers to expressions and selection index buffers). Instead the scan request is assembled
//! with a small **mutating builder**, [`VxScanBuilder`]: the caller constructs it, calls chained
//! `set_*` setters that mutate `&mut self` and return `()`, then calls a terminal `execute`
//! against a [`VxDataSource`](crate::data_source::ffi::VxDataSource) to obtain a [`VxScan`]. This
//! is the cleanest Diplomat idiom (mutate-then-build) and avoids a struct with raw pointers.
//!
//! The original scan executes on an async runtime. Diplomat has no `async` surface, so execution
//! is exposed as **synchronous** methods: each blocks on the shared FFI runtime internally
//! (`async` is bridged behind the method). `VxScan::next_partition` pulls the next
//! [`VxPartition`]; `VxPartition::next` pulls the next [`VxArray`]. Exhaustion — which the C ABI
//! signalled by returning NULL without an error — is represented as `Result<Option<..>, _>`
//! returning `Ok(None)`.

pub use ffi::VxEstimate;
pub use ffi::VxEstimateType;

#[diplomat::bridge]
pub mod ffi {
    use std::ops::Range;
    use std::sync::Arc;

    use futures::StreamExt;
    use vortex::array::stream::SendableArrayStream;
    use vortex::buffer::Buffer;
    use vortex::expr::root;
    use vortex::expr::stats::Precision;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::scan::DataSourceScan;
    use vortex::scan::Partition;
    use vortex::scan::PartitionStream;
    use vortex::scan::ScanRequest;
    use vortex::scan::selection::Selection;

    use crate::RUNTIME;
    use crate::array::ffi::VxArray;
    use crate::data_source::ffi::VxDataSource;
    use crate::dtype::ffi::VxDType;
    use crate::error::ffi::VortexFfiError;
    use crate::expression::ffi::VxExpr;

    /// How a [`VxScanBuilder`] row selection is interpreted.
    ///
    /// Mirrors the C `vx_scan_selection_include` enum.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VxSelectionInclude {
        /// Include all rows (the index list is ignored).
        All,
        /// Include only the rows at the given indices.
        IncludeRange,
        /// Exclude the rows at the given indices.
        ExcludeRange,
    }

    /// The precision of an estimate.
    ///
    /// Mirrors the C `vx_estimate_type` enum.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VxEstimateType {
        /// No estimate is available; `value` is meaningless.
        Unknown,
        /// `value` is exact.
        Exact,
        /// `value` is an upper bound.
        Inexact,
    }

    /// An estimated count (of partitions or rows), with an indication of its precision.
    ///
    /// Replaces the C `vx_estimate` out-parameter struct; Diplomat returns it by value. The
    /// `value` field is only meaningful when `kind` is not `Unknown`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct VxEstimate {
        /// The precision of `value`.
        pub kind: VxEstimateType,
        /// The estimated count. Meaningful only when `kind` is `Exact` or `Inexact`.
        pub value: u64,
    }

    /// A builder for a single scan request.
    ///
    /// Replaces the caller-populated C `vx_scan_options`/`vx_scan_selection` POD structs. All
    /// settings are optional: a freshly constructed builder scans all rows and columns. Call the
    /// `set_*` methods to configure, then `execute` against a data source.
    #[diplomat::opaque]
    pub struct VxScanBuilder(pub(crate) ScanRequest);

    /// A single traversal of a data source. A scan may be consumed only once.
    ///
    /// Replaces the C `vx_scan` opaque (a Pending/Started/Finished state machine). Pull
    /// partitions with [`Self::next_partition`].
    #[diplomat::opaque]
    pub struct VxScan(pub(crate) ScanState);

    /// An independent unit of scan work.
    ///
    /// Replaces the C `vx_partition` opaque (also a Pending/Started/Finished state machine).
    /// Pull arrays with [`Self::next`].
    #[diplomat::opaque]
    pub struct VxPartition(pub(crate) PartitionState);

    impl VxEstimate {
        /// An "unknown" estimate (no value available).
        pub fn unknown() -> VxEstimate {
            VxEstimate {
                kind: VxEstimateType::Unknown,
                value: 0,
            }
        }
    }

    impl VxScanBuilder {
        /// Create a scan builder that, unmodified, scans all rows and columns.
        ///
        /// Equivalent to a zero-initialised C `vx_scan_options`.
        #[diplomat::attr(auto, constructor)]
        pub fn new() -> Box<VxScanBuilder> {
            Box::new(VxScanBuilder(ScanRequest::default()))
        }

        /// Set the projection expression (which columns to return).
        ///
        /// Defaults to the root expression (all columns). Replaces `vx_scan_options.projection`.
        pub fn set_projection(&mut self, projection: &VxExpr) {
            self.0.projection = projection.inner().clone();
        }

        /// Set the predicate (filter) expression.
        ///
        /// Replaces `vx_scan_options.filter`.
        pub fn set_filter(&mut self, filter: &VxExpr) {
            self.0.filter = Some(filter.inner().clone());
        }

        /// Restrict the scan to the half-open row range `[begin, end)`.
        ///
        /// Replaces `vx_scan_options.row_range_begin`/`row_range_end`. A range of `[0, 0)`
        /// (or never calling this) means no row-range restriction.
        pub fn set_row_range(&mut self, begin: u64, end: u64) {
            self.0.row_range = (begin > 0 || end > 0).then_some(Range { start: begin, end });
        }

        /// Apply a row-index selection, replacing `vx_scan_options.selection`.
        ///
        /// When `include` is `All` the indices are ignored. For `IncludeRange`/`ExcludeRange`
        /// the given indices are copied into an owned buffer (the C ABI documented the same
        /// copy-and-may-free-after behaviour).
        pub fn set_selection(&mut self, include: VxSelectionInclude, indices: &[u64]) {
            self.0.selection = match include {
                VxSelectionInclude::All => Selection::All,
                VxSelectionInclude::IncludeRange => {
                    Selection::IncludeByIndex(Buffer::copy_from(indices))
                }
                VxSelectionInclude::ExcludeRange => {
                    Selection::ExcludeByIndex(Buffer::copy_from(indices))
                }
            };
        }

        /// Cap the number of rows returned. Replaces `vx_scan_options.limit` (0 means no limit).
        pub fn set_limit(&mut self, limit: u64) {
            self.0.limit = (limit != 0).then_some(limit);
        }

        /// Request results in storage order. Replaces `vx_scan_options.ordered`.
        pub fn set_ordered(&mut self, ordered: bool) {
            self.0.ordered = ordered;
        }

        /// Execute this builder against a data source, producing a scan.
        ///
        /// Terminal builder method. Replaces the C `vx_data_source_scan`. `async` execution is
        /// bridged internally by blocking on the shared FFI runtime. The estimated partition
        /// count — which the C ABI wrote into an optional out-parameter — is available afterward
        /// via [`VxScan::partition_count`].
        pub fn execute(
            &self,
            data_source: &VxDataSource,
        ) -> Result<Box<VxScan>, Box<VortexFfiError>> {
            let request = self.0.clone();
            let scan = RUNTIME
                .block_on(async move { data_source.inner().scan(request).await })
                .map_err(Box::<VortexFfiError>::from)?;
            Ok(Box::new(VxScan(ScanState::Pending(scan))))
        }
    }

    impl VxScan {
        /// The scan's output schema.
        ///
        /// Only available before the first call to [`Self::next_partition`]; afterward the scan
        /// has begun streaming and the dtype is no longer queryable (matching the C
        /// `vx_scan_dtype` contract). The returned dtype is an owned handle.
        #[diplomat::attr(auto, getter)]
        pub fn dtype(&self) -> Result<Box<VxDType>, Box<VortexFfiError>> {
            match &self.0 {
                ScanState::Pending(scan) => Ok(Box::new(VxDType(scan.dtype().clone()))),
                _ => Err(VortexFfiError::new(
                    "dtype unavailable: scan already started",
                )),
            }
        }

        /// The estimated number of partitions in this scan.
        ///
        /// Replaces the optional `estimate` out-parameter of `vx_data_source_scan`. Returns an
        /// `Unknown` estimate once the scan has started.
        #[diplomat::attr(auto, getter)]
        pub fn partition_count(&self) -> VxEstimate {
            match &self.0 {
                ScanState::Pending(scan) => {
                    VxEstimate::from_precision(scan.partition_count().map(|v| v as u64))
                }
                _ => VxEstimate::unknown(),
            }
        }

        /// Pull the next partition from the scan, or `None` when exhausted.
        ///
        /// Replaces `vx_scan_next_partition`; exhaustion (the C ABI's NULL-without-error) is
        /// `Ok(None)`. `async` is bridged internally. Not thread-safe: callers running a parallel
        /// pipeline must synchronise calls and hand each partition to a dedicated worker.
        pub fn next_partition(&mut self) -> Result<Option<Box<VxPartition>>, Box<VortexFfiError>> {
            let state = std::mem::replace(&mut self.0, ScanState::Finished);
            let mut stream = match state {
                ScanState::Pending(scan) => scan.partitions(),
                ScanState::Started(stream) => stream,
                ScanState::Finished => return Ok(None),
            };
            match RUNTIME.block_on(stream.next()) {
                Some(partition) => {
                    let partition = partition.map_err(Box::<VortexFfiError>::from)?;
                    self.0 = ScanState::Started(stream);
                    Ok(Some(Box::new(VxPartition(PartitionState::Pending(
                        partition,
                    )))))
                }
                None => {
                    self.0 = ScanState::Finished;
                    Ok(None)
                }
            }
        }
    }

    impl VxPartition {
        /// The partition's estimated row count.
        ///
        /// Replaces `vx_partition_row_count`. Must be read before the first [`Self::next`]; once
        /// the partition has started it returns an `Unknown` estimate.
        #[diplomat::attr(auto, getter)]
        pub fn row_count(&self) -> VxEstimate {
            match &self.0 {
                PartitionState::Pending(partition) => {
                    VxEstimate::from_precision(partition.row_count())
                }
                _ => VxEstimate::unknown(),
            }
        }

        /// Pull the next array from the partition, or `None` when exhausted.
        ///
        /// Replaces `vx_partition_next`; exhaustion is `Ok(None)`. `async` is bridged internally.
        /// Not thread-safe: call from a single thread per partition.
        pub fn next(&mut self) -> Result<Option<Box<VxArray>>, Box<VortexFfiError>> {
            let state = std::mem::replace(&mut self.0, PartitionState::Finished);
            let mut stream = match state {
                PartitionState::Pending(partition) => {
                    partition.execute().map_err(Box::<VortexFfiError>::from)?
                }
                PartitionState::Started(stream) => stream,
                PartitionState::Finished => return Ok(None),
            };
            match RUNTIME.block_on(stream.next()) {
                Some(array) => {
                    let array = array.map_err(Box::<VortexFfiError>::from)?;
                    self.0 = PartitionState::Started(stream);
                    Ok(Some(Box::new(VxArray(Arc::new(array)))))
                }
                None => {
                    self.0 = PartitionState::Finished;
                    Ok(None)
                }
            }
        }

        /// Consume the entire partition and export it as a single concatenated array.
        ///
        /// Replaces the C `vx_partition_scan_arrow`, which fully drained the partition into an
        /// `FFI_ArrowArrayStream` out-parameter. Diplomat has no surface for Arrow's C Data
        /// Interface stream struct, so instead of writing an `FFI_ArrowArrayStream`, this method
        /// blocks on the runtime to drain every chunk, concatenates them, and returns one owned
        /// [`VxArray`]; the caller can then bridge that array to Arrow via the array module's
        /// dedicated Arrow export. Consumes the partition: subsequent `next` calls return `None`.
        pub fn into_array(&mut self) -> Result<Box<VxArray>, Box<VortexFfiError>> {
            let state = std::mem::replace(&mut self.0, PartitionState::Finished);
            let stream = match state {
                PartitionState::Pending(partition) => {
                    partition.execute().map_err(Box::<VortexFfiError>::from)?
                }
                PartitionState::Started(stream) => stream,
                PartitionState::Finished => {
                    return Err(VortexFfiError::new(
                        "cannot consume partition: already consumed",
                    ));
                }
            };
            let chunks: Vec<_> = RUNTIME
                .block_on_stream(stream)
                .collect::<Result<_, _>>()
                .map_err(Box::<VortexFfiError>::from)?;
            let array = vortex::array::ArrayRef::concatenate(&chunks)
                .map_err(Box::<VortexFfiError>::from)?;
            Ok(Box::new(VxArray(Arc::new(array))))
        }
    }

    impl VxEstimate {
        /// Bridge a core [`Precision`] into a `VxEstimate`. Mirrors the C `write_estimate` helper.
        pub(crate) fn from_precision<T: Into<u64>>(estimate: Precision<T>) -> VxEstimate {
            match estimate {
                Precision::Exact(value) => VxEstimate {
                    kind: VxEstimateType::Exact,
                    value: value.into(),
                },
                Precision::Inexact(value) => VxEstimate {
                    kind: VxEstimateType::Inexact,
                    value: value.into(),
                },
                Precision::Absent => VxEstimate::unknown(),
            }
        }
    }

    /// Internal scan state machine, mirroring the C `VxScan` enum.
    pub enum ScanState {
        /// Not yet started; holds the un-driven scan.
        Pending(Box<dyn DataSourceScan>),
        /// Streaming partitions.
        Started(PartitionStream),
        /// Fully consumed.
        Finished,
    }

    /// Internal partition state machine, mirroring the C `VxPartitionScan` enum.
    pub enum PartitionState {
        /// Not yet started; holds the un-driven partition.
        Pending(Box<dyn Partition>),
        /// Streaming arrays.
        Started(SendableArrayStream),
        /// Fully consumed.
        Finished,
    }

    impl Default for ScanState {
        fn default() -> Self {
            ScanState::Finished
        }
    }
}
