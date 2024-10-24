use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{ready, Context, Poll};

use bytes::{Bytes, BytesMut};
use futures::Stream;
use futures_util::future::BoxFuture;
use futures_util::{stream, FutureExt, StreamExt, TryStreamExt};
use vortex::array::ChunkedArray;
use vortex::Array;
use vortex_dtype::DType;
use vortex_error::{vortex_err, vortex_panic, VortexError, VortexExpect, VortexResult};
use vortex_schema::Schema;

use crate::io::VortexReadAt;
use crate::layouts::read::cache::LayoutMessageCache;
use crate::layouts::read::selection::RowSelector;
use crate::layouts::read::{LayoutReader, MessageId, ReadResult};
use crate::layouts::RangeResult;
use crate::stream_writer::ByteRange;

pub struct LayoutBatchStream<R> {
    dtype: DType,
    row_count: u64,
    input: Option<R>,
    layout_reader: Box<dyn LayoutReader>,
    filter_reader: Option<Box<dyn LayoutReader>>,
    messages_cache: Arc<RwLock<LayoutMessageCache>>,
    current_selector: Option<RowSelector>,
    state: StreamingState<R>,
    offset: usize,
}

impl<R: VortexReadAt> LayoutBatchStream<R> {
    pub fn new(
        input: R,
        layout_reader: Box<dyn LayoutReader>,
        filter_reader: Option<Box<dyn LayoutReader>>,
        messages_cache: Arc<RwLock<LayoutMessageCache>>,
        dtype: DType,
        row_count: u64,
    ) -> Self {
        LayoutBatchStream {
            dtype,
            row_count,
            input: Some(input),
            layout_reader,
            filter_reader,
            messages_cache,
            current_selector: None,
            state: StreamingState::Init,
            offset: 0,
        }
    }

    pub fn schema(&self) -> Schema {
        Schema::new(self.dtype.clone())
    }

    fn store_messages(&self, messages: Vec<(MessageId, Bytes)>) {
        let mut write_cache_guard = self
            .messages_cache
            .write()
            .unwrap_or_else(|poison| vortex_panic!("Failed to write to message cache: {poison}"));
        for (message_id, buf) in messages {
            write_cache_guard.set(message_id, buf);
        }
    }
}

type StreamStateFuture<R> = BoxFuture<'static, VortexResult<(R, Vec<(MessageId, Bytes)>)>>;

#[derive(Debug, Clone, Copy)]
enum NextStreamState {
    Init,
    Filter(bool),
    Read(bool),
}

enum StreamingState<R> {
    Init,
    Filter(bool),
    Read(bool),
    Reading(StreamStateFuture<R>, NextStreamState),
    Error,
}

impl<R: VortexReadAt + Unpin + 'static> Stream for LayoutBatchStream<R> {
    type Item = VortexResult<Array>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                StreamingState::Init => {
                    let next_range = self
                        .filter_reader
                        .as_mut()
                        .map(|fr| fr.next_range())
                        .unwrap_or_else(|| self.layout_reader.as_mut().next_range())?;
                    match next_range {
                        RangeResult::ReadMore(messages) => {
                            let reader = self.input.take().ok_or_else(|| {
                                vortex_err!("Invalid state transition - reader dropped")
                            })?;
                            let read_future = read_ranges(reader, messages).boxed();
                            self.state =
                                StreamingState::Reading(read_future, NextStreamState::Init);
                        }
                        RangeResult::Rows(rs) => {
                            if let Some(s) = rs {
                                let read_more = s.end() as u64 != self.row_count;
                                self.current_selector = Some(s);
                                self.state = if self.filter_reader.is_some() {
                                    StreamingState::Filter(read_more)
                                } else {
                                    StreamingState::Read(read_more)
                                };
                            } else {
                                return Poll::Ready(None);
                            }
                        }
                    }
                }
                StreamingState::Read(read_more) => {
                    let read_more = *read_more;
                    let selector = self
                        .current_selector
                        .clone()
                        .vortex_expect("Must have asked for range");
                    if let Some(read) = self.layout_reader.read_next(selector)? {
                        match read {
                            ReadResult::ReadMore(messages) => {
                                let reader = self.input.take().ok_or_else(|| {
                                    vortex_err!("Invalid state transition - reader dropped")
                                })?;
                                let read_future = read_ranges(reader, messages).boxed();
                                self.state = StreamingState::Reading(
                                    read_future,
                                    NextStreamState::Read(read_more),
                                );
                            }
                            ReadResult::Batch(a) => return Poll::Ready(Some(Ok(a))),
                        }
                    } else if read_more {
                        self.state = StreamingState::Init;
                    } else {
                        return Poll::Ready(None);
                    }
                }
                StreamingState::Filter(read_more) => {
                    let read_more = *read_more;
                    let selector = self
                        .current_selector
                        .clone()
                        .vortex_expect("Must have asked for range");
                    if let Some(fr) = self
                        .filter_reader
                        .as_mut()
                        .vortex_expect("Can't filter without reader")
                        .read_next(selector)?
                    {
                        match fr {
                            ReadResult::ReadMore(messages) => {
                                let reader = self.input.take().ok_or_else(|| {
                                    vortex_err!("Invalid state transition - reader dropped")
                                })?;
                                let read_future = read_ranges(reader, messages).boxed();
                                self.state = StreamingState::Reading(
                                    read_future,
                                    NextStreamState::Filter(read_more),
                                );
                            }
                            ReadResult::Batch(b) => {
                                self.current_selector = Some(RowSelector::from_array(
                                    &b,
                                    self.offset,
                                    self.offset + b.len(),
                                )?);
                                self.offset += b.len();
                                self.state = StreamingState::Read(true);
                            }
                        }
                    } else {
                        self.state = StreamingState::Init;
                    }
                }
                StreamingState::Reading(f, next_state) => match ready!(f.poll_unpin(cx)) {
                    Ok((input, messages)) => {
                        let next_state = *next_state;
                        self.store_messages(messages);
                        self.input = Some(input);

                        self.state = match next_state {
                            NextStreamState::Init => StreamingState::Init,
                            NextStreamState::Filter(rm) => StreamingState::Filter(rm),
                            NextStreamState::Read(rm) => StreamingState::Read(rm),
                        };
                    }
                    Err(e) => {
                        self.state = StreamingState::Error;
                        return Poll::Ready(Some(Err(e)));
                    }
                },
                StreamingState::Error => return Poll::Ready(None),
            }
        }
    }
}

impl<R: VortexReadAt + Unpin + 'static> LayoutBatchStream<R> {
    pub async fn read_all(self) -> VortexResult<Array> {
        let dtype = self.schema().clone().into();
        let vecs: Vec<Array> = self.try_collect().await?;
        if vecs.len() == 1 {
            vecs.into_iter().next().ok_or_else(|| {
                vortex_panic!(
                    "Should be impossible: vecs.len() == 1 but couldn't get first element"
                )
            })
        } else {
            ChunkedArray::try_new(vecs, dtype).map(|e| e.into())
        }
    }
}

async fn read_ranges<R: VortexReadAt>(
    reader: R,
    ranges: Vec<(MessageId, ByteRange)>,
) -> VortexResult<(R, Vec<(MessageId, Bytes)>)> {
    stream::iter(ranges.into_iter())
        .map(|(id, range)| {
            let mut buf = BytesMut::with_capacity(range.len());
            unsafe { buf.set_len(range.len()) }

            let read_ft = reader.read_at_into(range.begin, buf);

            read_ft.map(|result| {
                result
                    .map(|res| (id, res.freeze()))
                    .map_err(VortexError::from)
            })
        })
        .buffered(10)
        .try_collect()
        .await
        .map(|b| (reader, b))
}
