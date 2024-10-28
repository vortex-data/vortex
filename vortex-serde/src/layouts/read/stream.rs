use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::task::{ready, Context, Poll};

use bytes::{Bytes, BytesMut};
use futures::Stream;
use futures_util::future::BoxFuture;
use futures_util::{stream, FutureExt, StreamExt, TryStreamExt};
use vortex_array::array::ChunkedArray;
use vortex_array::compute::filter;
use vortex_array::stats::ArrayStatistics;
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_error::{vortex_err, vortex_panic, VortexError, VortexExpect, VortexResult};
use vortex_schema::Schema;

use crate::io::VortexReadAt;
use crate::layouts::read::cache::LayoutMessageCache;
use crate::layouts::read::{LayoutReader, MessageId, ReadResult, Scan};
use crate::stream_writer::ByteRange;

pub struct LayoutBatchStream<R> {
    input: Option<R>,
    layout_reader: Box<dyn LayoutReader>,
    filter_reader: Option<Box<dyn LayoutReader>>,
    scan: Scan,
    messages_cache: Arc<RwLock<LayoutMessageCache>>,
    state: StreamingState<R>,
    dtype: DType,
    cached_mask: Option<Array>,
}

impl<R: VortexReadAt> LayoutBatchStream<R> {
    pub fn new(
        input: R,
        layout_reader: Box<dyn LayoutReader>,
        filter_reader: Option<Box<dyn LayoutReader>>,
        messages_cache: Arc<RwLock<LayoutMessageCache>>,
        dtype: DType,
        scan: Scan,
    ) -> Self {
        let state = if filter_reader.is_some() {
            StreamingState::FilterInit
        } else {
            StreamingState::Init
        };

        LayoutBatchStream {
            input: Some(input),
            layout_reader,
            filter_reader,
            scan,
            messages_cache,
            dtype,
            state,
            cached_mask: None,
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

#[derive(Default)]
enum StreamingState<R> {
    #[default]
    Init,
    FilterInit,
    Reading(StreamStateFuture<R>),
    FilterReading(StreamStateFuture<R>),
    Decoding(Array),
    Error,
}

impl<R: VortexReadAt + Unpin + Send + 'static> Stream for LayoutBatchStream<R> {
    type Item = VortexResult<Array>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                StreamingState::Init => {
                    if let Some(read) = self.layout_reader.read_next()? {
                        match read {
                            ReadResult::ReadMore(messages) => {
                                let reader = self.input.take().ok_or_else(|| {
                                    vortex_err!("Invalid state transition - reader dropped")
                                })?;
                                let read_future = read_ranges(reader, messages).boxed();
                                self.state = StreamingState::Reading(read_future);
                            }
                            ReadResult::Batch(a) => self.state = StreamingState::Decoding(a),
                        }
                    } else {
                        return Poll::Ready(None);
                    }
                }
                StreamingState::FilterInit => {
                    if let Some(read) = self
                        .filter_reader
                        .as_mut()
                        .vortex_expect("Can't filter without reader")
                        .read_next()?
                    {
                        match read {
                            ReadResult::ReadMore(messages) => {
                                let reader = self.input.take().ok_or_else(|| {
                                    vortex_err!("Invalid state transition - reader dropped")
                                })?;
                                let read_future = read_ranges(reader, messages).boxed();
                                self.state = StreamingState::FilterReading(read_future);
                            }
                            ReadResult::Batch(a) => {
                                let mask = self
                                    .scan
                                    .filter
                                    .as_ref()
                                    .vortex_expect("Cant filter without filter")
                                    .evaluate(&a)?;
                                self.cached_mask = Some(mask);
                                self.state = StreamingState::Init;
                            }
                        }
                    } else {
                        return Poll::Ready(None);
                    }
                }
                StreamingState::Decoding(arr) => {
                    let mut batch = arr.clone();

                    if let Some(mask) = self.cached_mask.take() {
                        if mask.statistics().compute_true_count().unwrap_or_default() == 0 {
                            self.state = StreamingState::Init;
                            continue;
                        }

                        batch = filter(batch, mask)?;
                    }

                    let goto_state = if self.filter_reader.is_some() {
                        StreamingState::FilterInit
                    } else {
                        StreamingState::Init
                    };
                    self.state = goto_state;
                    return Poll::Ready(Some(Ok(batch)));
                }
                StreamingState::Reading(f) => match ready!(f.poll_unpin(cx)) {
                    Ok((input, messages)) => {
                        self.store_messages(messages);
                        self.input = Some(input);

                        self.state = StreamingState::Init
                    }
                    Err(e) => {
                        self.state = StreamingState::Error;
                        return Poll::Ready(Some(Err(e)));
                    }
                },
                StreamingState::FilterReading(f) => match ready!(f.poll_unpin(cx)) {
                    Ok((input, messages)) => {
                        self.store_messages(messages);
                        self.input = Some(input);

                        self.state = StreamingState::FilterInit
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

impl<R: VortexReadAt + Unpin + Send + 'static> LayoutBatchStream<R> {
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
