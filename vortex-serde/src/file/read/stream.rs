use std::collections::{BTreeSet, VecDeque};
use std::mem;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{Context, Poll};

use bytes::{Bytes, BytesMut};
use futures::future::BoxFuture;
use futures::Stream;
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
use crate::file::read::{BatchRead, LayoutReader, MessageId, MessageLocator};
use crate::io::{Dispatch, IoDispatcher, VortexReadAt};

/// An asynchronous Vortex file that returns a [`Stream`] of [`Array`]s.
///
/// The file may be read from any source implementing [`VortexReadAt`], such
/// as memory, disk, and object storage.
///
/// Use [VortexReadBuilder][crate::file::read::builder::VortexReadBuilder] to build one
/// from a reader.
pub struct VortexFileArrayStream<R> {
    dtype: DType,
    row_count: u64,
    layout_reader: Box<dyn LayoutReader>,
    messages_cache: Arc<RwLock<LayoutMessageCache>>,
    splits: VecDeque<(usize, usize)>,
    row_mask: Option<RowMask>,
    state: Option<StreamingState>,
    input: R,
    dispatcher: IoDispatcher,
}

impl<R: VortexReadAt> VortexFileArrayStream<R> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input: R,
        layout_reader: Box<dyn LayoutReader>,
        filter_reader: Option<Box<dyn LayoutReader>>,
        messages_cache: Arc<RwLock<LayoutMessageCache>>,
        dtype: DType,
        row_count: u64,
        row_mask: Option<RowMask>,
        dispatcher: IoDispatcher,
    ) -> Self {
        VortexFileArrayStream {
            dtype,
            row_count,
            layout_reader,
            messages_cache,
            splits: VecDeque::new(),
            row_mask,
            state: Some(StreamingState::AddSplits(filter_reader)),
            input,
            dispatcher,
        }
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    pub fn schema(&self) -> Schema {
        Schema::new(self.dtype.clone())
    }

    fn store_messages(&self, messages: Vec<Message>) {
        let mut write_cache_guard = self
            .messages_cache
            .write()
            .unwrap_or_else(|poison| vortex_panic!("Failed to write to message cache: {poison}"));
        for Message(message_id, buf) in messages {
            write_cache_guard.set(message_id, buf);
        }
    }
}

/// A message that has had its bytes materialized onto the heap.
#[derive(Debug, Clone)]
struct Message(pub MessageId, pub Bytes);

type StreamMessages = Vec<Message>;
type StreamStateFuture = BoxFuture<'static, VortexResult<StreamMessages>>;

enum ReadingFor {
    Read(StreamStateFuture, RowMask, Option<LayoutReaderRef>),
    Filter(StreamStateFuture, RowMask, Box<dyn LayoutReader>),
}

enum ReadingPoll {
    Ready(StreamingState, StreamMessages),
    Pending(ReadingFor),
}

impl ReadingFor {
    fn future(&mut self) -> &mut StreamStateFuture {
        match self {
            ReadingFor::Read(future, ..) => future,
            ReadingFor::Filter(future, ..) => future,
        }
    }

    fn into_streaming_state(self) -> StreamingState {
        match self {
            ReadingFor::Read(.., row_mask, filter_reader) => {
                StreamingState::Read(row_mask, filter_reader)
            }
            ReadingFor::Filter(.., row_mask, reader) => StreamingState::Filter(row_mask, reader),
        }
    }

    fn poll_unpin(mut self, cx: &mut Context) -> VortexResult<ReadingPoll> {
        let messages = match self.future().poll_unpin(cx) {
            Poll::Pending => return Ok(ReadingPoll::Pending(self)),
            Poll::Ready(Err(err)) => return Err(err),
            Poll::Ready(Ok(x)) => x,
        };
        Ok(ReadingPoll::Ready(self.into_streaming_state(), messages))
    }
}

type LayoutReaderRef = Box<dyn LayoutReader>;

/// State of vortex file stream
///
/// The flow starts with `AddSplits` which produces all horizontal row splits in the file
/// Main read loop goes from `NextSplit` -> `Filter` (if there's filter) -> `Read`
/// `Filter` and `Read` states transition to `Reading` when they're blocked on an io operation which resumes back to
/// the previous state.
enum StreamingState {
    AddSplits(Option<LayoutReaderRef>),
    NextSplit(Option<LayoutReaderRef>),
    Filter(RowMask, LayoutReaderRef),
    Read(RowMask, Option<LayoutReaderRef>),
    Reading(ReadingFor),
    EndOfStream,
    Error,
}

enum StreamingTransition {
    GoTo(StreamingState),
    YieldTo(StreamingState),
    Produce(StreamingState, Array),
    Finished,
}

fn goto(next_state: StreamingState) -> VortexResult<StreamingTransition> {
    Ok(StreamingTransition::GoTo(next_state))
}

fn yield_to(next_state: StreamingState) -> VortexResult<StreamingTransition> {
    Ok(StreamingTransition::YieldTo(next_state))
}

fn produce(next_state: StreamingState, array: Array) -> VortexResult<StreamingTransition> {
    Ok(StreamingTransition::Produce(next_state, array))
}

fn finished() -> VortexResult<StreamingTransition> {
    Ok(StreamingTransition::Finished)
}

impl<R: VortexReadAt + Unpin> VortexFileArrayStream<R> {
    fn step(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        current_state: StreamingState,
    ) -> VortexResult<StreamingTransition> {
        match current_state {
            StreamingState::AddSplits(filter_reader) => {
                let mut splits = BTreeSet::new();
                splits.insert(self.row_count as usize);
                if let Some(filter_reader) = &filter_reader {
                    filter_reader.add_splits(0, &mut splits)?;
                }
                self.layout_reader.as_mut().add_splits(0, &mut splits)?;
                self.splits
                    .extend(splits.into_iter().tuple_windows::<(usize, usize)>());
                goto(StreamingState::NextSplit(filter_reader))
            }
            StreamingState::NextSplit(filter_reader) => {
                let Some((begin, end)) = self.splits.pop_front() else {
                    return finished();
                };

                let row_mask_removes_all_rows = self
                    .row_mask
                    .as_ref()
                    .map(|row_mask| row_mask.slice(begin, end).is_empty())
                    .unwrap_or(false);
                if row_mask_removes_all_rows {
                    return goto(StreamingState::NextSplit(filter_reader));
                }

                let mut split_mask = RowMask::new_valid_between(begin, end);
                match filter_reader {
                    Some(filter_reader) => goto(StreamingState::Filter(split_mask, filter_reader)),
                    None => {
                        if let Some(row_mask) = &self.row_mask {
                            split_mask.and_inplace(&row_mask.slice(begin, end))?;
                        };

                        goto(StreamingState::Read(split_mask, filter_reader))
                    }
                }
            }
            StreamingState::Filter(split_mask, mut filter_reader) => {
                let sel_begin = split_mask.begin();
                let sel_end = split_mask.end();

                match filter_reader.as_mut().read_selection(&split_mask)? {
                    Some(BatchRead::ReadMore(messages)) => goto(StreamingState::Reading(
                        ReadingFor::Filter(self.read_ranges(messages), split_mask, filter_reader),
                    )),
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
                            goto(StreamingState::NextSplit(Some(filter_reader)))
                        } else {
                            goto(StreamingState::Read(
                                RowMask::from_mask_array(&batch, sel_begin, sel_end)?,
                                Some(filter_reader),
                            ))
                        }
                    }
                    None => goto(StreamingState::NextSplit(Some(filter_reader))),
                }
            }
            StreamingState::Read(selector, filter_reader) => {
                match self.layout_reader.read_selection(&selector)? {
                    Some(BatchRead::ReadMore(message_ranges)) => {
                        let read_future = self.read_ranges(message_ranges);
                        goto(StreamingState::Reading(ReadingFor::Read(
                            read_future,
                            selector,
                            filter_reader,
                        )))
                    }
                    Some(BatchRead::Batch(array)) => {
                        produce(StreamingState::NextSplit(filter_reader), array)
                    }
                    None => goto(StreamingState::NextSplit(filter_reader)),
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

impl<R: VortexReadAt + Unpin> Stream for VortexFileArrayStream<R> {
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

impl<R: VortexReadAt + Unpin> VortexFileArrayStream<R> {
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
    ranges: Vec<MessageLocator>,
) -> VortexResult<Vec<Message>> {
    stream::iter(ranges.into_iter())
        .map(|MessageLocator(id, range)| {
            let mut buf = BytesMut::with_capacity(range.len());
            unsafe { buf.set_len(range.len()) }

            let read_ft = reader.read_at_into(range.begin, buf);

            read_ft.map(|result| {
                result
                    .map(|res| Message(id, res.freeze()))
                    .map_err(VortexError::from)
            })
        })
        .buffered(10)
        .try_collect()
        .await
}
