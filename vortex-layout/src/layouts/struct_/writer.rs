// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker, ready};

use async_trait::async_trait;
use futures::future::try_join_all;
use futures::task::{ArcWake, waker_ref};
use futures::{FutureExt, Stream, StreamExt, TryStreamExt};
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::{Array, ArrayContext, ToCanonical};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentSink;
use crate::sequence::{
    SendableSequentialStream, SequencePointer, SequentialStreamAdapter, SequentialStreamExt,
};
use crate::{IntoLayout as _, LayoutRef, LayoutStrategy};

pub struct StructStrategy<S> {
    child: S,
}

/// A [`LayoutStrategy`] that splits a StructArray batch into child layout writers
impl<S> StructStrategy<S>
where
    S: LayoutStrategy,
{
    pub fn new(child: S) -> Self {
        Self { child }
    }
}

#[async_trait]
impl<S> LayoutStrategy for StructStrategy<S>
where
    S: LayoutStrategy,
{
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        segment_sink: &dyn SegmentSink,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let Some(struct_dtype) = stream.dtype().as_struct_fields_opt().cloned() else {
            // nothing we can do if dtype is not struct
            return self
                .child
                .write_stream(ctx, segment_sink, stream, eof)
                .await;
        };
        if HashSet::<_, DefaultHashBuilder>::from_iter(struct_dtype.names().iter()).len()
            != struct_dtype.names().len()
        {
            vortex_bail!("StructLayout must have unique field names");
        }

        let stream = stream.map(|chunk| {
            let (sequence_id, chunk) = chunk?;
            if !chunk.all_valid() {
                vortex_bail!("Cannot push struct chunks with top level invalid values");
            };
            Ok((sequence_id, chunk))
        });

        // There are now fields so this is the layout leaf
        if struct_dtype.nfields() == 0 {
            let row_count = stream
                .try_fold(
                    0u64,
                    |acc, (_, arr)| async move { Ok(acc + arr.len() as u64) },
                )
                .await?;
            return Ok(StructLayout::new(row_count, dtype, vec![]).into_layout());
        }

        // stream<struct_chunk> -> stream<vec<column_chunk>>
        let columns_vec_stream = stream.map(|chunk| {
            let (sequence_id, chunk) = chunk?;
            let mut sequence_pointer = sequence_id.descend();
            let struct_chunk = chunk.to_struct();
            let columns: Vec<_> = (0..struct_chunk.struct_fields().nfields())
                .map(|idx| {
                    (
                        sequence_pointer.advance(),
                        struct_chunk.fields()[idx].to_array(),
                    )
                })
                .collect();
            Ok(columns)
        });

        // stream<vec<column_chunk>> -> vec<stream<column_chunk>>
        let column_streams = transpose_stream(columns_vec_stream, struct_dtype.nfields());

        let column_dtypes = (0..struct_dtype.nfields()).map(move |idx| {
            struct_dtype
                .field_by_index(idx)
                .vortex_expect("bound checked")
        });

        let layout_futures: Vec<_> = column_dtypes
            .zip_eq(column_streams)
            .map(move |(dtype, stream)| {
                let column_stream = SequentialStreamAdapter::new(dtype, stream).sendable();
                self.child
                    .write_stream(ctx, segment_sink, column_stream, eof.advance().descend())
                    .boxed()
            })
            .collect();

        let column_layouts = try_join_all(layout_futures).await?;
        // TODO(os): transposed stream could count row counts as well,
        // This must hold though, all columns must have the same row count of the struct layout
        let row_count = column_layouts.first().map(|l| l.row_count()).unwrap_or(0);
        Ok(StructLayout::new(row_count, dtype, column_layouts).into_layout())
    }
}

fn transpose_stream<T, S>(stream: S, elements: usize) -> Vec<impl Stream<Item = VortexResult<T>>>
where
    S: Stream<Item = VortexResult<Vec<T>>> + Unpin,
    T: Unpin + 'static,
{
    let state = Arc::new(Mutex::new(TransposeState {
        upstream: stream,
        buffers: (0..elements).map(|_| VecDeque::new()).collect(),
        exhausted: false,
    }));

    let shared_waker = Arc::new(SharedWaker {
        wakers: Default::default(),
    });

    (0..elements)
        .map(|index| TransposedStream {
            index,
            state: state.clone(),
            shared_waker: shared_waker.clone(),
        })
        .collect()
}

struct TransposeState<T, S>
where
    S: Stream<Item = VortexResult<Vec<T>>> + Unpin,
    T: Unpin,
{
    upstream: S,
    // TODO(os): make these buffers bounded so transposed streams can not run ahead unbounded
    buffers: Vec<VecDeque<VortexResult<T>>>,
    exhausted: bool,
}

struct SharedWaker {
    wakers: Arc<Mutex<HashMap<usize, Waker>>>,
}

impl SharedWaker {
    pub fn add(self: Arc<Self>, index: usize, waker: Waker) {
        self.wakers.lock().insert(index, waker);
    }
}

impl ArcWake for SharedWaker {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        for (_, waker) in arc_self.wakers.lock().drain() {
            waker.wake();
        }
    }
}

struct TransposedStream<T, S>
where
    S: Stream<Item = VortexResult<Vec<T>>> + Unpin,
    T: Unpin,
{
    index: usize,
    state: Arc<Mutex<TransposeState<T, S>>>,
    shared_waker: Arc<SharedWaker>,
}

impl<T, S> Stream for TransposedStream<T, S>
where
    S: Stream<Item = VortexResult<Vec<T>>> + Unpin,
    T: Unpin,
{
    type Item = VortexResult<T>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut guard = self.state.lock();
        if let Some(item) = guard.buffers[self.index].pop_front() {
            return Poll::Ready(Some(item));
        }

        // if we know upstream is exhausted we can skip polling it again.
        if guard.exhausted {
            return Poll::Ready(None);
        }

        self.shared_waker
            .clone()
            .add(self.index, cx.waker().clone());

        let shared_waker_ref = waker_ref(&self.shared_waker);
        let mut upstream_cx = Context::from_waker(&shared_waker_ref);
        match ready!(Pin::new(&mut guard.upstream).poll_next(&mut upstream_cx)) {
            None => {
                guard.exhausted = true;
                Poll::Ready(None)
            }
            Some(Ok(vec_t)) => {
                for (t, buffer) in vec_t.into_iter().zip_eq(guard.buffers.iter_mut()) {
                    buffer.push_back(Ok(t));
                }
                let item = guard.buffers[self.index]
                    .pop_front()
                    .vortex_expect("just pushed");
                Poll::Ready(Some(item))
            }
            Some(Err(err)) => {
                let shared_err = Arc::new(err);
                for buffer in guard.buffers.iter_mut() {
                    buffer.push_back(Err(shared_err.clone().into()));
                }
                Poll::Ready(Some(Err(shared_err.into())))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use vortex_array::arrays::{BoolArray, ChunkedArray, StructArray};
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayContext, Canonical, IntoArray as _};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, FieldNames, Nullability, PType};

    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::struct_::writer::StructStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::{SequenceId, SequentialArrayStreamExt};

    #[test]
    #[should_panic]
    fn fails_on_duplicate_field() {
        let strategy = StructStrategy::new(FlatLayoutStrategy::default());
        let (ptr, eof) = SequenceId::root().split();
        block_on(
            strategy.write_stream(
                &ArrayContext::empty(),
                &TestSegments::default(),
                Canonical::empty(&DType::Struct(
                    [
                        ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                        ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                    ]
                    .into_iter()
                    .collect(),
                    Nullability::NonNullable,
                ))
                .into_array()
                .to_array_stream()
                .sequenced(ptr),
                eof,
            ),
        )
        .unwrap();
    }

    #[test]
    fn fails_on_top_level_nulls() {
        let strategy = StructStrategy::new(FlatLayoutStrategy::default());
        let (ptr, eof) = SequenceId::root().split();
        let res = block_on(
            strategy.write_stream(
                &ArrayContext::empty(),
                &TestSegments::default(),
                StructArray::try_new(
                    ["a"].into(),
                    vec![buffer![1, 2, 3].into_array()],
                    3,
                    Validity::Array(BoolArray::from_iter(vec![true, true, false]).into_array()),
                )
                .unwrap()
                .into_array()
                .to_array_stream()
                .sequenced(ptr),
                eof,
            ),
        );
        assert!(
            format!("{}", res.unwrap_err())
                .starts_with("Cannot push struct chunks with top level invalid values"),
        )
    }

    #[test]
    fn write_empty_field_struct_array() {
        let strategy = StructStrategy::new(FlatLayoutStrategy::default());
        let (ptr, eof) = SequenceId::root().split();
        let res = block_on(
            strategy.write_stream(
                &ArrayContext::empty(),
                &TestSegments::default(),
                ChunkedArray::from_iter([
                    StructArray::try_new(FieldNames::default(), vec![], 3, Validity::NonNullable)
                        .unwrap()
                        .into_array(),
                    StructArray::try_new(FieldNames::default(), vec![], 5, Validity::NonNullable)
                        .unwrap()
                        .into_array(),
                ])
                .into_array()
                .to_array_stream()
                .sequenced(ptr),
                eof,
            ),
        );

        assert_eq!(res.unwrap().row_count(), 8);
    }
}
