// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;

use async_trait::async_trait;
use vortex_array::expr::Expression;
use vortex_array::stream::SendableArrayStream;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_layout::LayoutReaderRef;
use vortex_session::VortexSession;

use crate::ScanBuilder;
use crate::Selection;
use crate::api::DataSource;
use crate::api::DataSourceScan;
use crate::api::DataSourceScanRef;
use crate::api::Estimate;
use crate::api::ScanRequest;
use crate::api::Split;
use crate::api::SplitRef;

/// The default number of rows per Scan API split.
const DEFAULT_SPLIT_SIZE: u64 = 100_000;

/// An implementation of a [`DataSource`] that reads data from a [`LayoutReaderRef`].
pub struct LayoutReaderDataSource {
    reader: LayoutReaderRef,
    session: VortexSession,
    split_size: u64,
}

impl LayoutReaderDataSource {
    /// Creates a new [`LayoutReaderDataSource`].
    pub fn new(reader: LayoutReaderRef, session: VortexSession) -> Self {
        Self {
            reader,
            session,
            split_size: DEFAULT_SPLIT_SIZE,
        }
    }

    /// Sets the target number of rows per Scan API split.
    ///
    /// Each split drives a [`ScanBuilder`] over its row range, which internally handles
    /// physical layout alignment and I/O pipelining. This controls the engine-level
    /// parallelism granularity, not the I/O granularity.
    pub fn with_split_size(mut self, split_size: u64) -> Self {
        self.split_size = split_size;
        self
    }
}

#[async_trait]
impl DataSource for LayoutReaderDataSource {
    fn dtype(&self) -> &DType {
        self.reader.dtype()
    }

    fn row_count_estimate(&self) -> Estimate<u64> {
        Estimate::exact(self.reader.row_count())
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let total_rows = self.reader.row_count();
        let row_range = scan_request.row_range.unwrap_or(0..total_rows);

        let dtype = if let Some(proj) = &scan_request.projection {
            proj.return_dtype(self.reader.dtype())?
        } else {
            self.reader.dtype().clone()
        };

        Ok(Box::new(LayoutReaderScan {
            reader: self.reader.clone(),
            session: self.session.clone(),
            dtype,
            projection: scan_request.projection,
            filter: scan_request.filter,
            limit: scan_request.limit,
            selection: scan_request.selection,
            next_row: row_range.start,
            end_row: row_range.end,
            split_size: self.split_size,
        }))
    }

    fn deserialize_split(&self, _data: &[u8], _session: &VortexSession) -> VortexResult<SplitRef> {
        vortex_bail!("LayoutReader splits are not yet serializable");
    }
}

struct LayoutReaderScan {
    reader: LayoutReaderRef,
    session: VortexSession,
    dtype: DType,
    projection: Option<Expression>,
    filter: Option<Expression>,
    limit: Option<u64>,
    selection: Selection,
    next_row: u64,
    end_row: u64,
    split_size: u64,
}

#[async_trait]
impl DataSourceScan for LayoutReaderScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn remaining_splits_estimate(&self) -> Estimate<usize> {
        if self.next_row >= self.end_row {
            return Estimate::exact(0);
        }
        let remaining_rows = self.end_row - self.next_row;
        let splits = remaining_rows.div_ceil(self.split_size);
        Estimate {
            lower: 0,
            upper: Some(usize::try_from(splits).unwrap_or(usize::MAX)),
        }
    }

    async fn next_splits(&mut self, max_splits: usize) -> VortexResult<Vec<SplitRef>> {
        let mut splits = Vec::new();

        for _ in 0..max_splits {
            if self.next_row >= self.end_row {
                break;
            }

            if self.limit.is_some_and(|limit| limit == 0) {
                break;
            }

            let split_end = (self.next_row + self.split_size).min(self.end_row);
            let row_range = self.next_row..split_end;
            let split_rows = split_end - self.next_row;

            let split_limit = self.limit;
            if let Some(ref mut limit) = self.limit {
                *limit = limit.saturating_sub(split_rows);
            }

            splits.push(Box::new(LayoutReaderSplit {
                reader: self.reader.clone(),
                session: self.session.clone(),
                projection: self.projection.clone(),
                filter: self.filter.clone(),
                limit: split_limit,
                row_range,
                selection: self.selection.clone(),
            }) as SplitRef);

            self.next_row = split_end;
        }

        Ok(splits)
    }
}

struct LayoutReaderSplit {
    reader: LayoutReaderRef,
    session: VortexSession,
    projection: Option<Expression>,
    filter: Option<Expression>,
    limit: Option<u64>,
    row_range: Range<u64>,
    selection: Selection,
}

impl Split for LayoutReaderSplit {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        let mut builder = ScanBuilder::new(self.session, self.reader)
            .with_row_range(self.row_range)
            .with_selection(self.selection);

        if let Some(proj) = self.projection {
            builder = builder.with_projection(proj);
        }
        if let Some(filter) = self.filter {
            builder = builder.with_filter(filter);
        }
        if let Some(limit) = self.limit {
            builder = builder.with_limit(limit);
        }

        Ok(Box::pin(builder.into_array_stream()?))
    }

    fn row_count_estimate(&self) -> Estimate<u64> {
        Estimate {
            lower: 0,
            upper: Some(self.row_range.end - self.row_range.start),
        }
    }

    fn byte_size_estimate(&self) -> Estimate<u64> {
        Estimate::default()
    }
}
