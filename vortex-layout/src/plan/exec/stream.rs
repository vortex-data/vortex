// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::plan::PlanRef;
use crate::plan::exec::DecodeCache;
use crate::plan::exec::ExecGraph;
use crate::plan::exec::ExecOutput;
use crate::plan::exec::ExecState;
use crate::plan::exec::IoRequestId;
use crate::segments::SegmentSource;

/// Runs `plan` over `rows` of its row domain, restricted to `mask`, reading segments from
/// `source`, and yields the selected rows as a stream of arrays in row order.
///
/// Reads are issued as the graph discovers them and may complete in any order; the graph emits
/// in row order regardless. Every array yielded has at least one row, so a selection that keeps
/// nothing yields an empty stream.
///
/// `rows` must lie within the plan's row domain and `mask` must be as long as `rows`.
pub fn execute(
    session: VortexSession,
    plan: &PlanRef,
    rows: Range<u64>,
    mask: Mask,
    source: Arc<dyn SegmentSource>,
) -> VortexResult<ExecStream> {
    let graph = ExecGraph::try_new(session, plan, rows, mask, 0, DecodeCache::default())?;
    Ok(ExecStream {
        graph,
        source,
        inflight: FuturesUnordered::new(),
        done: false,
    })
}

type Delivery = BoxFuture<'static, (IoRequestId, VortexResult<BufferHandle>)>;

/// The arrays produced by [`execute`], in row order.
pub struct ExecStream {
    graph: ExecGraph,
    source: Arc<dyn SegmentSource>,
    inflight: FuturesUnordered<Delivery>,
    /// Whether the stream has ended, by finishing or by failing.
    done: bool,
}

impl ExecStream {
    /// Advances the graph until an array is ready, a read is pending, or the graph is done.
    fn step(&mut self, cx: &mut Context<'_>) -> Poll<Option<VortexResult<ArrayRef>>> {
        loop {
            match self.graph.state() {
                ExecState::Done => return Poll::Ready(None),
                ExecState::NeedsCompute => match self.graph.compute()? {
                    ExecOutput::Piece(array) => return Poll::Ready(Some(Ok(array))),
                    ExecOutput::NeedsIO(batch) => {
                        for request in batch {
                            let read = self.source.request(request.segment_id);
                            self.inflight
                                .push(async move { (request.id, read.await) }.boxed());
                        }
                    }
                    ExecOutput::Yield => {}
                },
                ExecState::Waiting => match self.inflight.poll_next_unpin(cx) {
                    Poll::Ready(Some((id, result))) => self.graph.set_io_result(id, result?)?,
                    Poll::Ready(None) => {
                        return Poll::Ready(Some(Err(vortex_err!(
                            "Exec graph waits with no reads in flight"
                        ))));
                    }
                    Poll::Pending => return Poll::Pending,
                },
            }
        }
    }
}

impl Stream for ExecStream {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.done {
            return Poll::Ready(None);
        }
        let next = this.step(cx);
        if matches!(next, Poll::Ready(None | Some(Err(_)))) {
            this.done = true;
        }
        next
    }
}
