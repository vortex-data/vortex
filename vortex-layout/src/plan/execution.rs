// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::segments::SegmentSource;

/// Future resolving to the array produced by a physical plan.
pub type PlanArrayFuture = BoxFuture<'static, VortexResult<ArrayRef>>;

/// Runtime dependencies shared by every node in a plan execution.
#[derive(Clone)]
pub struct PlanExecutionContext {
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
    row_offset: u64,
}

impl PlanExecutionContext {
    /// Creates an execution context over a segment source and Vortex session.
    pub fn new(segment_source: Arc<dyn SegmentSource>, session: VortexSession) -> Self {
        Self {
            segment_source,
            session,
            row_offset: 0,
        }
    }

    /// Sets the global row index of the first row in the root plan's row domain.
    pub fn with_row_offset(mut self, row_offset: u64) -> Self {
        self.row_offset = row_offset;
        self
    }

    /// Returns the global row index of the first row in the current plan's row domain.
    pub fn row_offset(&self) -> u64 {
        self.row_offset
    }

    /// Derives the execution context for a child whose row domain starts within this one.
    pub(crate) fn child_row_domain(&self, relative_row_offset: u64) -> VortexResult<Self> {
        let row_offset = self
            .row_offset
            .checked_add(relative_row_offset)
            .ok_or_else(|| vortex_error::vortex_err!("Plan row-domain offset overflow"))?;
        Ok(Self {
            segment_source: Arc::clone(&self.segment_source),
            session: self.session.clone(),
            row_offset,
        })
    }

    /// Returns the segment source used to satisfy leaf reads.
    pub fn segment_source(&self) -> &Arc<dyn SegmentSource> {
        &self.segment_source
    }

    /// Returns the Vortex session used for array decoding and expression execution.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}
