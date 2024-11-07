use std::collections::{BTreeSet, VecDeque};
use std::mem;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll};

use bytes::{Bytes, BytesMut};
use croaring::Bitmap;
use futures::Stream;
use futures_util::future::BoxFuture;
use futures_util::{stream, FutureExt, StreamExt, TryStreamExt};
use itertools::Itertools;
use vortex_array::array::ChunkedArray;
use vortex_array::compute::and;
use vortex_array::stats::ArrayStatistics;
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexError, VortexExpect, VortexResult};
use vortex_schema::Schema;

use crate::io::VortexReadAt;
use crate::layouts::read::cache::LayoutMessageCache;
use crate::layouts::read::mask::RowMask;
use crate::layouts::read::{BatchRead, LayoutReader, MessageId};
use crate::stream_writer::ByteRange;

/// Reads a layout from some memory, on-disk or elsewhere.
///
/// Instead of using [`LayoutBatchStream::new`], use a
/// [LayoutBatchStreamBuilder][crate::layouts::LayoutBatchStreamBuilder] to create an instance of
/// this struct.
pub struct LayoutBatchStream<R> {
    dtype: DType,
    row_count: u64,
    mask: Option<RowMask>,
    layout_reader: Box<dyn LayoutReader>,
    filter_reader: Option<Box<dyn LayoutReader>>,
    messages_cache: Arc<RwLock<LayoutMessageCache>>,
    splits: VecDeque<(usize, usize)>,
    state: StreamingState<R>,
}

impl<R: VortexReadAt> LayoutBatchStream<R> {
    pub fn new(
        input: R,
        mask: Option<RowMask>,
        layout_reader: Box<dyn LayoutReader>,
        filter_reader: Option<Box<dyn LayoutReader>>,
        messages_cache: Arc<RwLock<LayoutMessageCache>>,
        dtype: DType,
        row_count: u64,
    ) -> Self {
        LayoutBatchStream {
            dtype,
            row_count,
            mask,
            layout_reader,
            filter_reader,
            messages_cache,
            splits: VecDeque::new(),
            state: StreamingState::AddSplits(input),
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

type StreamMessages = Vec<(MessageId, Bytes)>;
type StreamStateFuture<R> = BoxFuture<'static, VortexResult<(R, StreamMessages)>>;

enum ReadingState<R> {
    BackToRead(StreamStateFuture<R>, RowMask),
    BackToFilter(StreamStateFuture<R>, RowMask, Box<dyn LayoutReader>),
}

enum ReadingPollState<R> {
    Ready(StreamingState<R>, StreamMessages),
    Pending(ReadingState<R>),
}

impl<R> ReadingState<R> {
    fn future(&mut self) -> &mut StreamStateFuture<R> {
        match self {
            ReadingState::BackToRead(future, ..) => future,
            ReadingState::BackToFilter(future, ..) => future,
        }
    }

    fn poll_unpin(mut self, cx: &mut Context) -> VortexResult<ReadingPollState<R>> {
        let (input, messages) = match self.future().poll_unpin(cx) {
            Poll::Pending => return Ok(ReadingPollState::Pending(self)),
            Poll::Ready(Err(err)) => return Err(err),
            Poll::Ready(Ok(x)) => x,
        };
        Ok(match self {
            ReadingState::BackToRead(.., mask) => {
                ReadingPollState::Ready(StreamingState::Read(input, mask), messages)
            }
            ReadingState::BackToFilter(.., mask, reader) => {
                ReadingPollState::Ready(StreamingState::Filter(input, mask, reader), messages)
            }
        })
    }
}

/// State of vortex file stream
///
/// The flow starts with `AddSplits` which produces all horizontal row splits in the file
/// Main read loop goes from `NextSplit` -> `Filter` (if there's filter) -> `Read`
/// `Filter` and `Read` states transition to `Reading` when they're blocked on an io operation which resumes back to
/// the previous state.
enum StreamingState<R> {
    AddSplits(R),
    NextSplit(R),
    Filter(R, RowMask, Box<dyn LayoutReader>),
    Read(R, RowMask),
    Reading(ReadingState<R>),
    Error,
    InPollNext,
}

impl<R: VortexReadAt + Unpin + 'static> Stream for LayoutBatchStream<R> {
    type Item = VortexResult<Array>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match mem::replace(&mut self.state, StreamingState::InPollNext) {
                StreamingState::AddSplits(input) => {
                    let mut splits = BTreeSet::new();
                    splits.insert(self.row_count as usize);
                    if let Some(filter_reader) = &self.filter_reader {
                        filter_reader.add_splits(0, &mut splits)?;
                    }
                    self.layout_reader.as_mut().add_splits(0, &mut splits)?;
                    self.splits
                        .extend(splits.into_iter().tuple_windows::<(usize, usize)>());
                    self.state = StreamingState::NextSplit(input);
                }
                StreamingState::NextSplit(input) => {
                    let Some((begin, end)) = self.splits.pop_front() else {
                        self.state = StreamingState::NextSplit(input);
                        return Poll::Ready(None);
                    };

                    let mut split_mask = unsafe {
                        RowMask::new_unchecked(
                            Bitmap::from_range(0..(end - begin) as u32),
                            begin,
                            end,
                        )
                    };

                    self.state = match mem::take(&mut self.filter_reader) {
                        Some(filter_reader) => {
                            StreamingState::Filter(input, split_mask, filter_reader)
                        }
                        None => {
                            if let Some(mask) = &self.mask {
                                split_mask.and_inplace(&mask.slice(begin, end))?;
                            };

                            StreamingState::Read(input, split_mask)
                        }
                    };
                }
                StreamingState::Filter(input, split_mask, mut filter_reader) => {
                    let sel_begin = split_mask.begin();
                    let sel_end = split_mask.end();

                    let Some(fr) = filter_reader.as_mut().read_selection(&split_mask)? else {
                        self.state = StreamingState::NextSplit(input);
                        continue;
                    };

                    match fr {
                        BatchRead::ReadMore(messages) => {
                            self.state = StreamingState::Reading(ReadingState::BackToFilter(
                                read_ranges(input, messages).boxed(),
                                split_mask,
                                filter_reader,
                            ));
                        }
                        BatchRead::Batch(mut batch) => {
                            if let Some(mask) = &self.mask {
                                batch = and(
                                    batch,
                                    mask.slice(sel_begin, sel_end).to_predicate_array()?,
                                )?;
                            }

                            self.filter_reader = Some(filter_reader);

                            if batch
                                .statistics()
                                .compute_true_count()
                                .vortex_expect("must be a bool array if it's a result of a filter")
                                == 0
                            {
                                self.state = StreamingState::NextSplit(input);
                                continue;
                            }
                            self.state = StreamingState::Read(
                                input,
                                RowMask::from_mask_array(&batch, sel_begin, sel_end)?,
                            );
                        }
                    }
                }
                StreamingState::Read(input, selector) => {
                    let Some(read) = self.layout_reader.read_selection(&selector)? else {
                        self.state = StreamingState::NextSplit(input);
                        continue;
                    };

                    match read {
                        BatchRead::ReadMore(messages) => {
                            let read_future = read_ranges(input, messages).boxed();
                            self.state = StreamingState::Reading(ReadingState::BackToRead(
                                read_future,
                                selector,
                            ));
                        }
                        BatchRead::Batch(a) => {
                            self.state = StreamingState::NextSplit(input);
                            return Poll::Ready(Some(Ok(a)));
                        }
                    }
                }
                StreamingState::Reading(reading_state) => {
                    match reading_state.poll_unpin(cx) {
                        Err(error) => {
                            self.state = StreamingState::Error;
                            return Poll::Ready(Some(Err(error)));
                        }
                        Ok(ReadingPollState::Pending(state)) => {
                            self.state = StreamingState::Reading(state);
                            return Poll::Pending;
                        }
                        Ok(ReadingPollState::Ready(state, messages)) => {
                            self.store_messages(messages);
                            self.state = state;
                        }
                    };
                }
                state @ StreamingState::Error => {
                    self.state = state;
                    return Poll::Ready(None);
                }
                StreamingState::InPollNext => {
                    vortex_panic!("unreachable")
                }
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
