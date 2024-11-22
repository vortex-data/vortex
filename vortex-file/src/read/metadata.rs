use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Context, Poll};

use futures::future::BoxFuture;
use futures::FutureExt as _;
use vortex_array::ArrayData;
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_io::{Dispatch as _, IoDispatcher, VortexReadAt};

use super::stream::{read_ranges, StreamMessages};
use super::{BatchRead, LayoutReader, MessageLocator};

pub struct MetadataReader<R: VortexReadAt> {
    input: R,
    dispatcher: Arc<IoDispatcher>,
    root_layout: Box<dyn LayoutReader>,
    state: State,
    stats: Vec<ArrayData>,
}

enum State {
    Initial,
    Reading(BoxFuture<'static, VortexResult<StreamMessages>>),
}

impl<R: VortexReadAt + Unpin> MetadataReader<R> {
    /// Schedule an asynchronous read of several byte ranges.
    ///
    /// IO is scheduled on the provided IO dispatcher.
    fn read_ranges(
        &self,
        ranges: Vec<MessageLocator>,
    ) -> BoxFuture<'static, VortexResult<StreamMessages>> {
        let reader = self.input.clone();

        let result_rx = self
            .dispatcher
            .dispatch(move || async move { read_ranges(reader, ranges).await })
            .vortex_expect("dispatch async task");

        result_rx
            .map(|res| match res {
                Ok(result) => result,
                Err(e) => vortex_bail!("dispatcher channel canceled: {e}"),
            })
            .boxed()
    }
}

impl<R: VortexReadAt + Unpin> Future for MetadataReader<R> {
    type Output = VortexResult<Option<Vec<ArrayData>>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.state {
            State::Initial => match self.root_layout.read_metadata()? {
                Some(batch_read) => match batch_read {
                    BatchRead::ReadMore(ranges) => {
                        let read_future = self.read_ranges(ranges);
                        self.get_mut().state = State::Reading(read_future);
                        return Poll::Pending;
                    }
                    BatchRead::Batch(array_data) => {
                        self.stats.push(array_data);
                        return Poll::Pending;
                    }
                },
                None => return Poll::Ready(Ok(Some(self.stats))),
            },
            State::Reading(f) => {
                let a = ready!(f.poll_unpin(cx));
                todo!()
            }
        }
    }
}
