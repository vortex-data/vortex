// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A shim [`ScanBuilder`] that mirrors the `vortex_scan::ScanBuilder` API but drives the v2
//! [`Scan`] state machine internally.

use std::collections::VecDeque;

use futures::stream::unfold;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::v2::layout::LayoutRef;
use crate::v2::scan::Scan;
use crate::v2::scan::ScanConfig;
use crate::v2::scan::planner::ComputeArgs;
use crate::v2::scan::scheduler::ScanAction;
use crate::v2::scan::scheduler::ScanEvent;
use crate::v2::selection::Selection;

/// A builder for configuring and executing a v2 layout scan.
///
/// Mirrors the API of `vortex_scan::ScanBuilder` to allow easy migration of existing code.
pub struct ScanBuilder {
    layout: LayoutRef,
    projection: Expression,
    filter: Option<Expression>,
    selection: Selection,
    config: ScanConfig,
    session: VortexSession,
}

impl ScanBuilder {
    /// Create a new scan builder for the given v2 layout.
    pub fn new(layout: LayoutRef, session: VortexSession) -> Self {
        Self {
            layout,
            projection: root(),
            filter: None,
            selection: Selection::All,
            config: ScanConfig::default(),
            session,
        }
    }

    /// Set the filter expression for the scan.
    pub fn with_filter(mut self, filter: Expression) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set an optional filter expression.
    pub fn with_some_filter(mut self, filter: Option<Expression>) -> Self {
        self.filter = filter;
        self
    }

    /// Set the projection expression for the scan.
    pub fn with_projection(mut self, projection: Expression) -> Self {
        self.projection = projection;
        self
    }

    /// Select specific row indices to include.
    pub fn with_row_indices(mut self, row_indices: Buffer<u64>) -> Self {
        self.selection = Selection::IncludeByIndex(row_indices);
        self
    }

    /// Set the row selection.
    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    /// Returns the [`DType`] of the scan output after applying the projection.
    pub fn dtype(&self) -> VortexResult<DType> {
        self.projection.return_dtype(self.layout.dtype())
    }

    /// Build and return an [`ArrayStream`] that drives the v2 scan state machine.
    ///
    /// Segment reads are performed sequentially. For parallel I/O, use the [`Scan`] state
    /// machine directly.
    pub fn into_array_stream(self) -> VortexResult<SendableArrayStream> {
        let dtype = self.dtype()?;
        let scan = Scan::try_new(
            &self.layout,
            &self.projection,
            self.filter.as_ref(),
            &self.selection,
            self.config,
            &self.session,
        )?;
        let stream = unfold(ScanStreamState::new(scan), |state| async move {
            state.next_item().await
        });
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }
}

/// Internal state for the scan stream driver.
struct ScanStreamState {
    scan: Scan,
    pending_emits: VecDeque<ArrayRef>,
    done: bool,
}

impl ScanStreamState {
    fn new(scan: Scan) -> Self {
        Self {
            scan,
            pending_emits: VecDeque::new(),
            done: false,
        }
    }

    /// Drives the scan state machine until the next array is ready to yield, or the scan completes.
    async fn next_item(mut self) -> Option<(VortexResult<ArrayRef>, Self)> {
        loop {
            // Drain pending emits first.
            if let Some(array) = self.pending_emits.pop_front() {
                return Some((Ok(array), self));
            }

            if self.done {
                return None;
            }

            // Get the next batch of actions from the scan state machine.
            let actions = match self.scan.actions() {
                Ok(actions) => actions,
                Err(e) => return Some((Err(e), self)),
            };

            if actions.is_empty() {
                return None;
            }

            for action in actions {
                match action {
                    ScanAction::ReadSegment {
                        read_id,
                        source,
                        segment_id,
                    } => match source.request(segment_id).await {
                        Ok(buffer_handle) => match buffer_handle.try_into_host_sync() {
                            Ok(byte_buffer) => {
                                if let Err(e) = self.scan.post_event(ScanEvent::SegmentReady {
                                    read_id,
                                    buffer: byte_buffer,
                                }) {
                                    return Some((Err(e), self));
                                }
                            }
                            Err(e) => {
                                if let Err(e2) = self
                                    .scan
                                    .post_event(ScanEvent::SegmentFailed { read_id, error: e })
                                {
                                    return Some((Err(e2), self));
                                }
                            }
                        },
                        Err(e) => {
                            if let Err(e2) = self
                                .scan
                                .post_event(ScanEvent::SegmentFailed { read_id, error: e })
                            {
                                return Some((Err(e2), self));
                            }
                        }
                    },
                    ScanAction::Compute {
                        compute_id,
                        compute,
                        segments,
                        inputs,
                    } => match compute(ComputeArgs { segments, inputs }) {
                        Ok(result) => {
                            if let Err(e) = self
                                .scan
                                .post_event(ScanEvent::ComputeReady { compute_id, result })
                            {
                                return Some((Err(e), self));
                            }
                        }
                        Err(error) => {
                            if let Err(e) = self
                                .scan
                                .post_event(ScanEvent::ComputeFailed { compute_id, error })
                            {
                                return Some((Err(e), self));
                            }
                        }
                    },
                    ScanAction::Emit { result, .. } => {
                        if let Some(array) = result {
                            self.pending_emits.push_back(array);
                        }
                    }
                    ScanAction::Done => {
                        self.done = true;
                    }
                }
            }
        }
    }
}
