// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`DataSource`] implementation backed by the v2 layout scan state machine.

use std::any::Any;

use async_trait::async_trait;
use futures::stream;
use futures::stream::StreamExt;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::stats::Precision;
use vortex_array::scalar::Scalar;
use vortex_array::stats::StatsSet;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_scan::DataSource;
use vortex_scan::DataSourceScan;
use vortex_scan::DataSourceScanRef;
use vortex_scan::Partition;
use vortex_scan::PartitionRef;
use vortex_scan::PartitionStream;
use vortex_scan::ScanRequest;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::v2;
use crate::v2::layout::LayoutRef;

/// Convert a [`Selection`] to a [`v2::selection::Selection`].
///
/// The two types are structurally identical but live in different crates, so we need a manual
/// conversion.
fn convert_selection(selection: Selection) -> v2::selection::Selection {
    match selection {
        Selection::All => v2::selection::Selection::All,
        Selection::IncludeByIndex(buf) => v2::selection::Selection::IncludeByIndex(buf),
        Selection::ExcludeByIndex(buf) => v2::selection::Selection::ExcludeByIndex(buf),
        Selection::IncludeRoaring(r) => v2::selection::Selection::IncludeRoaring(r),
        Selection::ExcludeRoaring(r) => v2::selection::Selection::ExcludeRoaring(r),
    }
}

/// A [`DataSource`] implementation that reads data using the v2 layout scan state machine.
pub struct V2LayoutDataSource {
    layout: LayoutRef,
    session: VortexSession,
}

impl V2LayoutDataSource {
    /// Creates a new [`V2LayoutDataSource`].
    pub fn new(layout: LayoutRef, session: VortexSession) -> Self {
        Self { layout, session }
    }
}

#[async_trait]
impl DataSource for V2LayoutDataSource {
    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> Option<Precision<u64>> {
        Some(Precision::exact(self.layout.row_count()))
    }

    fn byte_size(&self) -> Option<Precision<u64>> {
        None
    }

    fn deserialize_partition(
        &self,
        _data: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<PartitionRef> {
        vortex_bail!("V2LayoutDataSource partitions are not yet serializable");
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let dtype = scan_request.projection.return_dtype(self.layout.dtype())?;

        // If the dtype is an empty struct, and there is no filter, return a length-only scan.
        if let DType::Struct(fields, Nullability::NonNullable) = &dtype
            && fields.nfields() == 0
            && scan_request.filter.is_none()
        {
            let row_count = self.layout.row_count();
            let row_count = scan_request.selection.row_count(row_count);
            let row_count = scan_request.limit.map_or(row_count, |l| row_count.min(l));

            return Ok(Box::new(Empty { dtype, row_count }));
        }

        Ok(Box::new(V2LayoutScan {
            layout: self.layout.clone(),
            session: self.session.clone(),
            dtype,
            projection: scan_request.projection,
            filter: scan_request.filter,
            selection: scan_request.selection,
            limit: scan_request.limit,
        }))
    }

    async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
        Ok(StatsSet::default())
    }
}

/// A scan over a single v2 layout, producing exactly one partition.
struct V2LayoutScan {
    layout: LayoutRef,
    session: VortexSession,
    dtype: DType,
    projection: Expression,
    filter: Option<Expression>,
    selection: Selection,
    limit: Option<u64>,
}

impl DataSourceScan for V2LayoutScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Option<Precision<usize>> {
        Some(Precision::exact(1usize))
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        let partition = Box::new(V2LayoutPartition {
            layout: self.layout,
            session: self.session,
            projection: self.projection,
            filter: self.filter,
            selection: self.selection,
            limit: self.limit,
        }) as PartitionRef;
        stream::iter([Ok(partition)]).boxed()
    }
}

/// A single partition that drives the v2 scan state machine.
struct V2LayoutPartition {
    layout: LayoutRef,
    session: VortexSession,
    projection: Expression,
    filter: Option<Expression>,
    selection: Selection,
    limit: Option<u64>,
}

impl Partition for V2LayoutPartition {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn row_count(&self) -> Option<Precision<u64>> {
        let row_count = self.selection.row_count(self.layout.row_count());
        let row_count = self.limit.map_or(row_count, |l| row_count.min(l));

        Some(if self.filter.is_some() {
            Precision::inexact(row_count)
        } else {
            Precision::exact(row_count)
        })
    }

    fn byte_size(&self) -> Option<Precision<u64>> {
        None
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        // TODO(v2): push limit into the V2 scan state machine once supported.
        let v2_selection = convert_selection(self.selection);

        let builder = v2::scan::shim::ScanBuilder::new(self.layout, self.session)
            .with_projection(self.projection)
            .with_some_filter(self.filter)
            .with_selection(v2_selection);

        builder.into_array_stream()
    }
}

/// A scan that produces no data, only empty arrays with the correct row count.
struct Empty {
    dtype: DType,
    row_count: u64,
}

impl DataSourceScan for Empty {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn partition_count(&self) -> Option<Precision<usize>> {
        Some(Precision::exact(1usize))
    }

    fn partitions(self: Box<Self>) -> PartitionStream {
        stream::iter([Ok(self as _)]).boxed()
    }
}

impl Partition for Empty {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn row_count(&self) -> Option<Precision<u64>> {
        Some(Precision::exact(self.row_count))
    }

    fn byte_size(&self) -> Option<Precision<u64>> {
        Some(Precision::exact(0u64))
    }

    fn execute(mut self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let scalar = Scalar::default_value(&self.dtype);
        let dtype = self.dtype.clone();

        let iter = std::iter::from_fn(move || {
            if self.row_count == 0 {
                return None;
            }
            let chunk_size = usize::try_from(self.row_count).unwrap_or(usize::MAX);
            self.row_count -= chunk_size as u64;
            Some(VortexResult::Ok(
                ConstantArray::new(scalar.clone(), chunk_size).into_array(),
            ))
        });

        Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
            dtype,
            stream::iter(iter),
        )))
    }
}
