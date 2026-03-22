// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_layout::v2::reader::ReaderRef;
use vortex_session::VortexSession;

use crate::Selection;

pub struct ScanBuilder2 {
    reader: ReaderRef,
    projection: Expression,
    filter: Option<Expression>,
    limit: Option<u64>,
    row_range: Range<u64>,
    row_selection: Selection, // NOTE: applies to the selected row range.
    session: VortexSession,
}

impl ScanBuilder2 {
    pub fn new(reader: ReaderRef, session: VortexSession) -> Self {
        let row_range = 0..reader.row_count();
        Self {
            reader,
            projection: root(),
            filter: None,
            limit: None,
            row_range,
            row_selection: Selection::All,
            session,
        }
    }

    pub fn with_filter(mut self, filter: Expression) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_some_filter(mut self, filter: Option<Expression>) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_projection(mut self, projection: Expression) -> Self {
        self.projection = projection;
        self
    }

    pub fn with_limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn with_row_range(mut self, row_range: Range<u64>) -> Self {
        self.row_range = row_range;
        self
    }

    /// Sets the row selection to use the given selection (relative to the row range).
    pub fn with_row_selection(mut self, row_selection: Selection) -> Self {
        self.row_selection = row_selection;
        self
    }

    /// Sets the row selection to include only the given row indices (relative to the row range).
    pub fn with_row_indices(mut self, row_indices: Buffer<u64>) -> Self {
        self.row_selection = Selection::IncludeByIndex(row_indices);
        self
    }

    pub fn into_array_stream(self) -> VortexResult<impl ArrayStream> {
        let projection = self.projection.optimize_recursive(self.reader.dtype())?;
        let filter = self
            .filter
            .map(|f| f.optimize_recursive(self.reader.dtype()))
            .transpose()?;

        let dtype = projection.return_dtype(self.reader.dtype())?;

        // So we wrap the reader for filtering.
        let filter_reader = filter
            .as_ref()
            .map(|f| self.reader.clone().apply(f))
            .transpose()?;
        let projection_reader = self.reader.clone().apply(&projection)?;

        // TODO(ngates): wrap filter in `falsify` expression for pruning.

        let reader_stream = self.reader.project(self.row_range)?;

        Ok(Scan {
            dtype,
            stream: todo!("construct scan stream"),
        })
    }
}

struct Scan {
    dtype: DType,
    stream: SendableArrayStream,
}

impl ArrayStream for Scan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl Stream for Scan {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!()
    }
}
