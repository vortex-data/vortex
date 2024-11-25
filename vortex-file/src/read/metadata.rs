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
use super::{LayoutMessageCache, LayoutReader, MessageLocator, MetadataRead};
use crate::read::stream::Message;

pub struct MetadataFetcher<R: VortexReadAt> {
    input: R,
    dispatcher: Arc<IoDispatcher>,
    root_layout: Box<dyn LayoutReader>,
    layout_cache: Arc<RwLock<LayoutMessageCache>>,
    state: State,
}

enum State {
    Initial,
    Reading(BoxFuture<'static, VortexResult<StreamMessages>>),
}

impl<R: VortexReadAt + Unpin> MetadataFetcher<R> {
    pub fn fetch(
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

impl<R: VortexReadAt + Unpin> Future for MetadataFetcher<R> {
    type Output = VortexResult<Option<Vec<ArrayData>>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        println!("MetadataFetcher::poll");

        loop {
            match &mut self.state {
                State::Initial => match self.root_layout.read_metadata()? {
                    MetadataRead::ReadMore(messages) => {
                        let read_future = self.read_ranges(messages);
                        self.state = State::Reading(read_future);
                    }
                    MetadataRead::Batches(array_data) => {
                        return Poll::Ready(Ok(Some(array_data)));
                    }
                    MetadataRead::None => {
                        return Poll::Ready(Ok(None));
                    }
                },
                State::Reading(ref mut f) => {
                    println!("State::Reading");
                    let messages = ready!(f.poll_unpin(cx))?;
                    println!("State::ready");
                    match self.layout_cache.write() {
                        Ok(mut cache) => {
                            for Message(message_id, bytes) in messages.into_iter() {
                                cache.set(message_id, bytes);
                            }
                        }
                        Err(poison) => {
                            vortex_panic!("Failed to write to message cache: {poison}")
                        }
                    }

                    self.state = State::Initial;
                }
            }
        }
    }
}
