use std::collections::VecDeque;
use std::io;
use std::io::ErrorKind;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll, Waker};

use futures::Stream;
use futures_util::future::BoxFuture;
use futures_util::{FutureExt, StreamExt};
use vortex_array::ArrayData;
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};
use vortex_io::{Dispatch, IoDispatcher, VortexReadAt, VortexReadRanges};

use crate::{LayoutMessageCache, LayoutReader, Message, MessageLocator, PollRead, RowMask};

const NUM_TO_COALESCE: usize = 8;

pub(crate) trait ReadMasked {
    type Value;

    /// Read a Layout into a `V`, applying the given bitmask. Only entries corresponding to positions
    /// where mask is `true` will be included in the output.
    fn read_masked(&self, mask: &RowMask) -> VortexResult<Option<PollRead<Self::Value>>>;
}

/// Read an array with a [`RowMask`].
pub(crate) struct ReadArray {
    layout: Arc<dyn LayoutReader>,
}

impl ReadArray {
    pub(crate) fn new(layout: Arc<dyn LayoutReader>) -> Self {
        Self { layout }
    }
}

impl ReadMasked for ReadArray {
    type Value = ArrayData;

    /// Read given mask out of the reader
    fn read_masked(&self, mask: &RowMask) -> VortexResult<Option<PollRead<ArrayData>>> {
        self.layout.poll_read(mask)
    }
}

enum RowMaskState<V> {
    Pending(RowMask),
    Ready(V),
    Empty,
}

pub struct BufferedLayoutReader<R, S, V, RM> {
    /// Stream of row masks to read
    read_masks: S,
    row_mask_reader: RM,
    in_flight: Option<BoxFuture<'static, io::Result<Vec<Message>>>>,
    queued: VecDeque<RowMaskState<V>>,
    io_read: VortexReadRanges<R>,
    dispatcher: Arc<IoDispatcher>,
    cache: Arc<RwLock<LayoutMessageCache>>,
}

impl<R, S, V, RM> BufferedLayoutReader<R, S, V, RM>
where
    R: VortexReadAt,
    S: Stream<Item = VortexResult<RowMask>> + Unpin,
    RM: ReadMasked<Value = V>,
{
    pub fn new(
        read: R,
        dispatcher: Arc<IoDispatcher>,
        read_masks: S,
        row_mask_reader: RM,
        cache: Arc<RwLock<LayoutMessageCache>>,
    ) -> Self {
        Self {
            read_masks,
            row_mask_reader,
            in_flight: None,
            queued: VecDeque::new(),
            io_read: VortexReadRanges::new(read, dispatcher.clone(), 1 << 20),
            dispatcher,
            cache,
        }
    }

    fn store_messages(&self, messages: Vec<Message>) {
        let mut write_cache_guard = self
            .cache
            .write()
            .unwrap_or_else(|poison| vortex_panic!("Failed to write to message cache: {poison}"));
        for Message(message_id, buf) in messages {
            write_cache_guard.set(message_id, buf);
        }
    }

    fn gather_read_messages(
        &mut self,
        cx: &mut Context<'_>,
    ) -> VortexResult<(Vec<MessageLocator>, bool)> {
        let mut to_read = Vec::with_capacity(NUM_TO_COALESCE);
        let mut read_more_count = 0;

        // Poll all queued pending masks to see if we can make progress
        for queued_res in self.queued.iter_mut() {
            match queued_res {
                RowMaskState::Pending(pending_mask) => {
                    if let Some(pending_read) = self.row_mask_reader.read_masked(pending_mask)? {
                        match pending_read {
                            PollRead::ReadMore(m) => {
                                to_read.extend(m);
                                read_more_count += 1;
                            }
                            PollRead::Value(v) => {
                                *queued_res = RowMaskState::Ready(v);
                            }
                        }
                    } else {
                        *queued_res = RowMaskState::Empty;
                    }
                }
                RowMaskState::Ready(_) => {}
                RowMaskState::Empty => {}
            }
        }

        let mut exhausted = false;
        while read_more_count < NUM_TO_COALESCE {
            match self.read_masks.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(next_mask))) => {
                    if let Some(read_result) = self.row_mask_reader.read_masked(&next_mask)? {
                        match read_result {
                            PollRead::ReadMore(m) => {
                                self.queued.push_back(RowMaskState::Pending(next_mask));
                                to_read.extend(m);
                                read_more_count += 1;
                            }
                            PollRead::Value(v) => {
                                self.queued.push_back(RowMaskState::Ready(v));
                            }
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    return Err(e);
                }
                Poll::Ready(None) => {
                    exhausted = true;
                    break;
                }
                Poll::Pending => {
                    break;
                }
            }
        }
        Ok((to_read, exhausted))
    }

    fn dispatch_messages(
        &self,
        messages: Vec<MessageLocator>,
        waker: Waker,
    ) -> BoxFuture<'static, io::Result<Vec<Message>>> {
        let reader = self.io_read.clone();
        self.dispatcher
            .dispatch(move || async move {
                let read_messages = reader
                    .read_byte_ranges(messages.iter().map(|msg| msg.1.as_range()).collect())
                    .map(move |read_res| {
                        Ok(messages
                            .into_iter()
                            .map(|loc| loc.0)
                            .zip(read_res?)
                            .map(|(loc, bytes)| Message(loc, bytes))
                            .collect())
                    })
                    .await;
                waker.wake();
                read_messages
            })
            .vortex_expect("Async task dispatch")
            .map(|res| res.unwrap_or_else(|e| Err(io::Error::new(ErrorKind::Other, e))))
            .boxed()
    }
}

impl<R, S, V, RM> Stream for BufferedLayoutReader<R, S, V, RM>
where
    R: VortexReadAt + Unpin,
    S: Stream<Item = VortexResult<RowMask>> + Unpin,
    RM: ReadMasked<Value = V> + Unpin,
    V: Unpin,
{
    type Item = VortexResult<V>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let exhausted = if let Some(in_flight) = &mut self.in_flight {
            match in_flight.poll_unpin(cx) {
                Poll::Ready(msgs) => {
                    self.store_messages(
                        msgs.map_err(|e| vortex_err!("Cancelled in flight read {e}"))?,
                    );
                    let (messages, exhausted) = self.gather_read_messages(cx)?;
                    if !messages.is_empty() {
                        self.in_flight = Some(self.dispatch_messages(messages, cx.waker().clone()));
                    } else {
                        self.in_flight = None;
                    }
                    exhausted
                }
                // If read is pending see if we have any available results
                Poll::Pending => false,
            }
        } else {
            let (messages, exhausted) = self.gather_read_messages(cx)?;
            if !messages.is_empty() {
                self.in_flight = Some(self.dispatch_messages(messages, cx.waker().clone()));
            }
            exhausted
        };

        while let Some(next_mask) = self.queued.pop_front() {
            match next_mask {
                RowMaskState::Pending(m) => {
                    self.queued.push_front(RowMaskState::Pending(m));
                    return Poll::Pending;
                }
                RowMaskState::Ready(next_ready) => return Poll::Ready(Some(Ok(next_ready))),
                RowMaskState::Empty => continue,
            }
        }

        if exhausted {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}
