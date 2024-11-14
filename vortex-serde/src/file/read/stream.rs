use std::collections::{BTreeSet, VecDeque};
use std::mem;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll};

use bytes::{Bytes, BytesMut};
use futures::Stream;
use futures_util::future::BoxFuture;
use futures_util::{stream, FutureExt, StreamExt, TryStreamExt};
use itertools::Itertools;
use vortex_array::array::ChunkedArray;
use vortex_array::compute::and_kleene;
use vortex_array::stats::ArrayStatistics;
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_panic, VortexError, VortexExpect, VortexResult};
use vortex_schema::Schema;

use crate::file::read::cache::LayoutMessageCache;
use crate::file::read::mask::RowMask;
use crate::file::read::{BatchRead, LayoutReader, MessageId};
use crate::io::VortexReadAt;
use crate::stream_writer::ByteRange;

/// Reads a layout from some memory, on-disk or elsewhere.
///
/// Instead of using [`VortexFileArrayStream::new`], use a
/// [VortexReadBuilder][crate::file::read::builder::VortexReadBuilder] to create an instance of
/// this struct.
pub struct VortexFileArrayStream<R> {
    dtype: DType,
    row_count: u64,
    layout_reader: Box<dyn LayoutReader>,
    messages_cache: Arc<RwLock<LayoutMessageCache>>,
    splits: VecDeque<(usize, usize)>,
    row_mask: Option<RowMask>,
    state: Option<StreamingState<R>>,
}

impl<R: VortexReadAt> VortexFileArrayStream<R> {
    pub fn new(
        input: R,
        layout_reader: Box<dyn LayoutReader>,
        filter_reader: Option<Box<dyn LayoutReader>>,
        messages_cache: Arc<RwLock<LayoutMessageCache>>,
        dtype: DType,
        row_count: u64,
        row_mask: Option<RowMask>,
    ) -> Self {
        VortexFileArrayStream {
            dtype,
            row_count,
            layout_reader,
            messages_cache,
            splits: VecDeque::new(),
            row_mask,
            state: Some(StreamingState::AddSplits(input, filter_reader)),
        }
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
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

enum ReadingFor<R> {
    Read(StreamStateFuture<R>, RowMask, Option<LayoutReaderRef>),
    Filter(StreamStateFuture<R>, RowMask, Box<dyn LayoutReader>),
}

enum ReadingPoll<R> {
    Ready(StreamingState<R>, StreamMessages),
    Pending(ReadingFor<R>),
}

impl<R> ReadingFor<R> {
    fn future(&mut self) -> &mut StreamStateFuture<R> {
        match self {
            ReadingFor::Read(future, ..) => future,
            ReadingFor::Filter(future, ..) => future,
        }
    }

    fn into_streaming_state(self, input: R) -> StreamingState<R> {
        match self {
            ReadingFor::Read(.., row_mask, filter_reader) => {
                StreamingState::Read(input, row_mask, filter_reader)
            }
            ReadingFor::Filter(.., row_mask, reader) => {
                StreamingState::Filter(input, row_mask, reader)
            }
        }
    }

    fn poll_unpin(mut self, cx: &mut Context) -> VortexResult<ReadingPoll<R>> {
        let (input, messages) = match self.future().poll_unpin(cx) {
            Poll::Pending => return Ok(ReadingPoll::Pending(self)),
            Poll::Ready(Err(err)) => return Err(err),
            Poll::Ready(Ok(x)) => x,
        };
        Ok(ReadingPoll::Ready(
            self.into_streaming_state(input),
            messages,
        ))
    }
}

type LayoutReaderRef = Box<dyn LayoutReader>;

/// State of vortex file stream
///
/// The flow starts with `AddSplits` which produces all horizontal row splits in the file
/// Main read loop goes from `NextSplit` -> `Filter` (if there's filter) -> `Read`
/// `Filter` and `Read` states transition to `Reading` when they're blocked on an io operation which resumes back to
/// the previous state.
enum StreamingState<R> {
    AddSplits(R, Option<LayoutReaderRef>),
    NextSplit(R, Option<LayoutReaderRef>),
    Filter(R, RowMask, LayoutReaderRef),
    Read(R, RowMask, Option<LayoutReaderRef>),
    Reading(ReadingFor<R>),
    EndOfStream,
    Error,
}

enum StreamingTransition<R> {
    GoTo(StreamingState<R>),
    YieldTo(StreamingState<R>),
    Produce(StreamingState<R>, Array),
    Finished,
}

fn goto<R>(next_state: StreamingState<R>) -> VortexResult<StreamingTransition<R>> {
    Ok(StreamingTransition::GoTo(next_state))
}

fn yield_to<R>(next_state: StreamingState<R>) -> VortexResult<StreamingTransition<R>> {
    Ok(StreamingTransition::YieldTo(next_state))
}

fn produce<R>(next_state: StreamingState<R>, array: Array) -> VortexResult<StreamingTransition<R>> {
    Ok(StreamingTransition::Produce(next_state, array))
}

fn finished<R>() -> VortexResult<StreamingTransition<R>> {
    Ok(StreamingTransition::Finished)
}

impl<R: VortexReadAt + Unpin + 'static> VortexFileArrayStream<R> {
    fn step(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        current_state: StreamingState<R>,
    ) -> VortexResult<StreamingTransition<R>> {
        match current_state {
            StreamingState::AddSplits(input, filter_reader) => {
                let mut splits = BTreeSet::new();
                splits.insert(self.row_count as usize);
                if let Some(filter_reader) = &filter_reader {
                    filter_reader.add_splits(0, &mut splits)?;
                }
                self.layout_reader.as_mut().add_splits(0, &mut splits)?;
                self.splits
                    .extend(splits.into_iter().tuple_windows::<(usize, usize)>());
                goto(StreamingState::NextSplit(input, filter_reader))
            }
            StreamingState::NextSplit(input, filter_reader) => {
                let Some((begin, end)) = self.splits.pop_front() else {
                    return finished();
                };

                let row_mask_removes_all_rows = self
                    .row_mask
                    .as_ref()
                    .map(|row_mask| row_mask.slice(begin, end).is_empty())
                    .unwrap_or(false);
                if row_mask_removes_all_rows {
                    return goto(StreamingState::NextSplit(input, filter_reader));
                }

                let mut split_mask = RowMask::new_valid_between(begin, end);
                match filter_reader {
                    Some(filter_reader) => {
                        goto(StreamingState::Filter(input, split_mask, filter_reader))
                    }
                    None => {
                        if let Some(row_mask) = &self.row_mask {
                            split_mask.and_inplace(&row_mask.slice(begin, end))?;
                        };

                        goto(StreamingState::Read(input, split_mask, filter_reader))
                    }
                }
            }
            StreamingState::Filter(input, split_mask, mut filter_reader) => {
                let sel_begin = split_mask.begin();
                let sel_end = split_mask.end();

                match filter_reader.as_mut().read_selection(&split_mask)? {
                    Some(BatchRead::ReadMore(messages)) => {
                        goto(StreamingState::Reading(ReadingFor::Filter(
                            read_ranges(input, messages).boxed(),
                            split_mask,
                            filter_reader,
                        )))
                    }
                    Some(BatchRead::Batch(mut batch)) => {
                        if let Some(row_mask) = &self.row_mask {
                            // Either `and` or `and_kleene` is fine. They only differ on `false AND
                            // null`, but RowMask::from_mask_array only cares which values are true.
                            batch = and_kleene(
                                batch,
                                row_mask.slice(sel_begin, sel_end).to_mask_array()?,
                            )?;
                        }

                        if batch
                            .statistics()
                            .compute_true_count()
                            .vortex_expect("must be a bool array if it's a result of a filter")
                            == 0
                        {
                            goto(StreamingState::NextSplit(input, Some(filter_reader)))
                        } else {
                            goto(StreamingState::Read(
                                input,
                                RowMask::from_mask_array(&batch, sel_begin, sel_end)?,
                                Some(filter_reader),
                            ))
                        }
                    }
                    None => goto(StreamingState::NextSplit(input, Some(filter_reader))),
                }
            }
            StreamingState::Read(input, selector, filter_reader) => {
                match self.layout_reader.read_selection(&selector)? {
                    Some(BatchRead::ReadMore(messages)) => {
                        let read_future = read_ranges(input, messages).boxed();
                        goto(StreamingState::Reading(ReadingFor::Read(
                            read_future,
                            selector,
                            filter_reader,
                        )))
                    }
                    Some(BatchRead::Batch(array)) => {
                        produce(StreamingState::NextSplit(input, filter_reader), array)
                    }
                    None => goto(StreamingState::NextSplit(input, filter_reader)),
                }
            }
            StreamingState::Reading(reading_state) => match reading_state.poll_unpin(cx)? {
                ReadingPoll::Pending(reading_state) => {
                    yield_to(StreamingState::Reading(reading_state))
                }
                ReadingPoll::Ready(next_state, messages) => {
                    self.store_messages(messages);
                    goto(next_state)
                }
            },
            StreamingState::Error => vortex_bail!("you polled a stream that previously erred"),
            StreamingState::EndOfStream => finished(),
        }
    }
}

impl<R: VortexReadAt + Unpin + 'static> Stream for VortexFileArrayStream<R> {
    type Item = VortexResult<Array>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let Some(current_state) = mem::take(&mut self.state) else {
                vortex_panic!("called poll_next while in poll_next");
            };

            match self.as_mut().step(cx, current_state) {
                Ok(StreamingTransition::GoTo(next_state)) => self.state = Some(next_state),
                Ok(StreamingTransition::YieldTo(next_state)) => {
                    self.state = Some(next_state);
                    return Poll::Pending;
                }
                Ok(StreamingTransition::Produce(next_state, array)) => {
                    self.state = Some(next_state);
                    return Poll::Ready(Some(Ok(array)));
                }
                Ok(StreamingTransition::Finished) => {
                    self.state = Some(StreamingState::EndOfStream);
                    return Poll::Ready(None);
                }
                Err(error) => {
                    self.state = Some(StreamingState::Error);
                    return Poll::Ready(Some(Err(error)));
                }
            }
        }
    }
}

impl<R: VortexReadAt + Unpin + 'static> VortexFileArrayStream<R> {
    pub async fn read_all(self) -> VortexResult<Array> {
        let dtype = self.dtype().clone();
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
