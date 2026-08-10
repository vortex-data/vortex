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
}

impl PlanExecutionContext {
    /// Creates an execution context over a segment source and Vortex session.
    pub fn new(segment_source: Arc<dyn SegmentSource>, session: VortexSession) -> Self {
        Self {
            segment_source,
            session,
        }
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
