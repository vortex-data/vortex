// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::expr::Expression;
use vortex_array::expr::root;
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
    row_selection: Selection,
    session: VortexSession,
}

impl ScanBuilder2 {
    pub fn new(reader: ReaderRef, session: VortexSession) -> Self {
        Self {
            reader,
            projection: root(),
            filter: None,
            limit: None,
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

    pub fn with_row_selection(mut self, row_selection: Selection) -> Self {
        self.row_selection = row_selection;
        self
    }

    pub fn with_row_indices(mut self, row_indices: Buffer<u64>) -> Self {
        self.row_selection = Selection::IncludeByIndex(row_indices);
        self
    }

    pub fn into_array_stream(self) -> VortexResult<SendableArrayStream> {
        todo!()
    }
}

struct Scan {}
