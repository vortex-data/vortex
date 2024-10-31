use std::collections::{BTreeSet, VecDeque};
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{ready, Context, Poll};

use bytes::{Bytes, BytesMut};
use croaring::Bitmap;
use futures::Stream;
use futures_util::future::BoxFuture;
use futures_util::{stream, FutureExt, StreamExt, TryStreamExt};
use itertools::Itertools;
use vortex::array::ChunkedArray;
use vortex::Array;
use vortex_dtype::DType;
use vortex_error::{vortex_err, vortex_panic, VortexError, VortexExpect, VortexResult};
use vortex_schema::Schema;

use crate::io::VortexReadAt;
use crate::layouts::read::cache::LayoutMessageCache;
use crate::layouts::read::mask::RowMask;
use crate::layouts::read::{LayoutReader, MessageId, ReadResult};
use crate::stream_writer::ByteRange;

pub struct LayoutBatchStream<R> {
    dtype: DType,
    row_count: u64,
    input: Option<R>,
    layout_reader: Box<dyn LayoutReader>,
    filter_reader: Option<Box<dyn LayoutReader>>,
    messages_cache: Arc<RwLock<LayoutMessageCache>>,
    splits: VecDeque<(usize, usize)>,
    current_selector: Option<RowMask>,
    state: StreamingState<R>,
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
            splits: VecDeque::new(),
            current_selector: None,
            state: StreamingState::AddSplits,
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

enum StreamingState<R> {
    AddSplits,
    NextSplit,
    Filter,
    Read,
    Reading(StreamStateFuture<R>, bool),
    Error,
}

impl<R: VortexReadAt + Unpin + 'static> Stream for LayoutBatchStream<R> {
    type Item = VortexResult<Array>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                StreamingState::Read => {
                    let selector = self
                        .current_selector
                        .clone()
                        .vortex_expect("Must have asked for range");
                    if let Some(read) = self.layout_reader.read_selection(selector)? {
                        match read {
                            ReadResult::ReadMore(messages) => {
                                let reader = self.input.take().ok_or_else(|| {
                                    vortex_err!("Invalid state transition - reader dropped")
                                })?;
                                let read_future = read_ranges(reader, messages).boxed();
                                self.state = StreamingState::Reading(read_future, false);
                            }
                            ReadResult::Batch(a) => {
                                self.state = StreamingState::NextSplit;
                                return Poll::Ready(Some(Ok(a)));
                            }
                        }
                    } else {
                        self.state = StreamingState::NextSplit;
                    }
                }
                StreamingState::Filter => {
                    let selector = self
                        .current_selector
                        .clone()
                        .vortex_expect("Must have asked for range");
                    let sel_begin = selector.begin();
                    let sel_end = selector.end();
                    if let Some(fr) = self
                        .filter_reader
                        .as_mut()
                        .vortex_expect("Can't filter without reader")
                        .read_selection(selector)?
                    {
                        match fr {
                            ReadResult::ReadMore(messages) => {
                                let reader = self.input.take().ok_or_else(|| {
                                    vortex_err!("Invalid state transition - reader dropped")
                                })?;
                                let read_future = read_ranges(reader, messages).boxed();
                                self.state = StreamingState::Reading(read_future, true);
                            }
                            ReadResult::Batch(b) => {
                                self.current_selector =
                                    Some(RowMask::from_array(&b, sel_begin, sel_end)?);
                                self.state = StreamingState::Read;
                            }
                        }
                    } else {
                        self.state = StreamingState::NextSplit;
                    }
                }
                StreamingState::Reading(f, filter_more) => match ready!(f.poll_unpin(cx)) {
                    Ok((input, messages)) => {
                        let filter_more = *filter_more;
                        self.store_messages(messages);
                        self.input = Some(input);

                        self.state = if filter_more {
                            StreamingState::Filter
                        } else {
                            StreamingState::Read
                        };
                    }
                    Err(e) => {
                        self.state = StreamingState::Error;
                        return Poll::Ready(Some(Err(e)));
                    }
                },
                StreamingState::AddSplits => {
                    let mut splits = BTreeSet::new();
                    splits.insert(self.row_count as usize);
                    self.filter_reader
                        .as_mut()
                        .map(|fr| fr.add_splits(0, &mut splits))
                        .unwrap_or_else(|| self.layout_reader.as_mut().add_splits(0, &mut splits));
                    self.splits
                        .extend(splits.into_iter().tuple_windows::<(usize, usize)>());
                    self.state = StreamingState::NextSplit;
                }
                StreamingState::NextSplit => {
                    self.current_selector = self.splits.pop_front().map(|(begin, end)| {
                        RowMask::new(Bitmap::from_range(0..(end - begin) as u32), begin, end)
                    });

                    if self.current_selector.is_none() {
                        return Poll::Ready(None);
                    }

                    self.state = if self.filter_reader.is_some() {
                        StreamingState::Filter
                    } else {
                        StreamingState::Read
                    };
                }
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
