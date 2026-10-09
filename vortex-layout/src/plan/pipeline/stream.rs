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
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::ReadId;
use super::Scan;
use super::Split;
use super::Turn;
use crate::plan::PlanRef;
use crate::segments::SegmentSource;

/// Runs `plan` over `rows` of its row domain, restricted to `mask`, reading segments from
/// `source`, and yields the selected rows as a stream of arrays in row order.
///
/// Reads are issued as the pipelines ask for them and may complete in any order. Every array
/// yielded has at least one row.
pub fn execute(
    session: VortexSession,
    plan: &PlanRef,
    rows: Range<u64>,
    mask: Mask,
    source: Arc<dyn SegmentSource>,
) -> VortexResult<ScanStream> {
    let scan = Scan::try_new(session, plan.clone(), vec![Split { rows, mask }])?;
    Ok(ScanStream::new(scan, source))
}

type Delivery = BoxFuture<'static, (ReadId, VortexResult<BufferHandle>)>;

/// A [`Scan`]'s output as a stream, its reads answered by a [`SegmentSource`]. Arrays of
/// different splits interleave; each split's arrays are in row order.
pub struct ScanStream {
    scan: Scan,
    source: Arc<dyn SegmentSource>,
    inflight: FuturesUnordered<Delivery>,
    done: bool,
}

impl ScanStream {
    /// Streams `scan`, reading from `source`.
    pub fn new(scan: Scan, source: Arc<dyn SegmentSource>) -> Self {
        Self {
            scan,
            source,
            inflight: FuturesUnordered::new(),
            done: false,
        }
    }

    fn step(&mut self, cx: &mut Context<'_>) -> Poll<Option<VortexResult<ArrayRef>>> {
        loop {
            match self.scan.step()? {
                Turn::Output(_, array) => return Poll::Ready(Some(Ok(array))),
                Turn::Read(read) => {
                    let bytes = self.source.request(read.segment_id);
                    self.inflight
                        .push(async move { (read.id, bytes.await) }.boxed());
                }
                Turn::Waiting => match self.inflight.poll_next_unpin(cx) {
                    Poll::Ready(Some((id, bytes))) => self.scan.deliver(id, bytes?)?,
                    Poll::Ready(None) => {
                        return Poll::Ready(Some(Err(vortex_error::vortex_err!(
                            "Scan waits with no reads in flight"
                        ))));
                    }
                    Poll::Pending => return Poll::Pending,
                },
                Turn::Done => return Poll::Ready(None),
            }
        }
    }
}

impl Stream for ScanStream {
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
