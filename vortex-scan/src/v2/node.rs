// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::StreamExecRef;
use std::sync::Arc;
use vortex_error::VortexResult;

pub type StreamNodeRef = Arc<dyn StreamNode>;

/// A logical node in the Vortex stream processing graph.
pub trait StreamNode: 'static + Send + Sync {
    /// The total number of rows represented by this node.
    fn row_count(&self) -> u64;

    /// Executes the stream node, returning a [`StreamExecRef`].
    ///
    // TODO(ngates): we may want to take a RowRange here to support partitioning?
    fn execute(&self) -> VortexResult<StreamExecRef>;
}
