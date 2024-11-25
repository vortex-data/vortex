use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{ready, Context, Poll};

use futures::future::BoxFuture;
use futures::FutureExt as _;
use vortex_array::ArrayData;
use vortex_error::{vortex_bail, vortex_panic, VortexExpect as _, VortexResult};
use vortex_io::{Dispatch as _, IoDispatcher, VortexReadAt};

use super::stream::{read_ranges, StreamMessages};
use super::{BatchRead, LayoutMessageCache, LayoutReader, MessageLocator};
use crate::read::stream::Message;

pub struct MetadataReader<R: VortexReadAt> {
    input: R,
    dispatcher: Arc<IoDispatcher>,
    root_layout: Box<dyn LayoutReader>,
    layout_cache: Arc<RwLock<LayoutMessageCache>>,
    state: State,
    metadata_table: Vec<ArrayData>,
}

enum State {
    Initial,
    Reading(BoxFuture<'static, VortexResult<StreamMessages>>),
}

impl<R: VortexReadAt + Unpin> MetadataReader<R> {
    pub fn new(
        input: R,
        dispatcher: Arc<IoDispatcher>,
        root_layout: Box<dyn LayoutReader>,
        layout_cache: Arc<RwLock<LayoutMessageCache>>,
    ) -> Self {
        Self {
            input,
            dispatcher,
            root_layout,
            layout_cache,
            state: State::Initial,
            metadata_table: Vec::default(),
        }
    }
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

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match &mut self.state {
            State::Initial => match self.root_layout.read_metadata()? {
                Some(batch_read) => match batch_read {
                    BatchRead::ReadMore(ranges) => {
                        let read_future = self.read_ranges(ranges);
                        self.state = State::Reading(read_future);
                        Poll::Pending
                    }
                    BatchRead::Batch(array_data) => {
                        self.metadata_table.push(array_data);
                        Poll::Pending
                    }
                },
                None => Poll::Ready(Ok(Some(std::mem::take(&mut self.metadata_table)))),
            },
            State::Reading(ref mut f) => {
                let messages = ready!(f.poll_unpin(cx))?;
                let mut write_cache_guard = self.layout_cache.write().unwrap_or_else(|poison| {
                    vortex_panic!("Failed to write to message cache: {poison}")
                });

                for Message(message_id, bytes) in messages.into_iter() {
                    write_cache_guard.set(message_id, bytes);
                }
                drop(write_cache_guard);
                self.state = State::Initial;
                Poll::Pending
            }
        }
    }
}
