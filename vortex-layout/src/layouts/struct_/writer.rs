use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use arcref::ArcRef;
use futures::future::try_join_all;
use futures::{Stream, StreamExt};
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::{ArrayContext, ToCanonical};
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail, vortex_err};

use crate::layouts::struct_::StructLayout;
use crate::segments::SequenceWriter;
use crate::{
    IntoLayout as _, LayoutStrategy, SendableLayoutWriter, SendableSequentialStream,
    SequentialStreamAdapter, SequentialStreamExt,
};

pub struct StructStrategy {
    child: ArcRef<dyn LayoutStrategy>,
}

/// A [`LayoutStrategy`] that splits a StructArray batch into child layout writers
impl StructStrategy {
    pub fn new(child: ArcRef<dyn LayoutStrategy>) -> Self {
        Self { child }
    }
}

impl LayoutStrategy for StructStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> SendableLayoutWriter {
        let dtype = stream.dtype().clone();
        let Some(struct_dtype) = stream.dtype().as_struct().cloned() else {
            // nothing we can do if dtype is not struct
            return self.child.write_stream(ctx, sequence_writer, stream);
        };
        if HashSet::from_iter(struct_dtype.names().iter()).len() != struct_dtype.names().len() {
            return Box::pin(async {
                Err(vortex_err!("StructLayout must have unique field names"))
            });
        }

        let stream = stream.map(|chunk| {
            let (sequence_id, chunk) = chunk?;
            if !chunk.all_valid()? {
                vortex_bail!("Cannot push struct chunks with top level invalid values");
            };
            Ok((sequence_id, chunk))
        });

        // stream<struct_chunk> -> stream<vec<column_chunk>>
        let columns_vec_stream = stream.map(|chunk| {
            let (sequence_id, chunk) = chunk?;
            let mut sequence_pointer = sequence_id.descend();
            let struct_chunk = chunk.to_struct()?;
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
        let child = self.child.clone();
        let ctx = ctx.clone();
        let layout_futures = column_dtypes
            .zip_eq(column_streams)
            .map(move |(dtype, stream)| {
                let column_stream = SequentialStreamAdapter::new(dtype, stream).sendable();
                child.write_stream(&ctx, sequence_writer.clone(), column_stream)
            });

        Box::pin(async move {
            let column_layouts = try_join_all(layout_futures).await?;
            // TODO(os): transposed stream could count row counts as well,
            // This must hold though, all columns must have the same row count of the struct layout
            let row_count = column_layouts.first().map(|l| l.row_count()).unwrap_or(0);
            Ok(StructLayout::new(row_count, dtype, column_layouts).into_layout())
        })
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
    (0..elements)
        .map(|index| TransposedStream {
            index,
            state: state.clone(),
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

struct TransposedStream<T, S>
where
    S: Stream<Item = VortexResult<Vec<T>>> + Unpin,
    T: Unpin,
{
    index: usize,
    state: Arc<Mutex<TransposeState<T, S>>>,
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

        // support non fused streams
        if guard.exhausted {
            return Poll::Ready(None);
        }

        match ready!(Pin::new(&mut guard.upstream).poll_next(cx)) {
            None => {
                guard.exhausted = true;
                Poll::Ready(None)
            }
            Some(Ok(vec_t)) => {
                for (t, buffer) in vec_t.into_iter().zip_eq(guard.buffers.iter_mut()) {
                    buffer.push_back(Ok(t));
                }
                Poll::Ready(Some(
                    guard.buffers[self.index]
                        .pop_front()
                        .vortex_expect("just pushed"),
                ))
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
    use std::sync::Arc;

    use arcref::ArcRef;
    use futures::executor::block_on;
    use futures::stream;
    use vortex_array::arrays::{BoolArray, StructArray};
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayContext, IntoArray as _};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::struct_::writer::StructStrategy;
    use crate::segments::{SequenceWriter, TestSegments};
    use crate::sequence::SequenceId;
    use crate::{LayoutStrategy, SequentialStreamAdapter, SequentialStreamExt};

    #[test]
    #[should_panic]
    fn fails_on_duplicate_field() {
        let strategy =
            StructStrategy::new(ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())));
        block_on(
            strategy.write_stream(
                &ArrayContext::empty(),
                SequenceWriter::new(Box::new(TestSegments::default())),
                SequentialStreamAdapter::new(
                    DType::Struct(
                        Arc::new(
                            [
                                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                            ]
                            .into_iter()
                            .collect(),
                        ),
                        Nullability::NonNullable,
                    ),
                    stream::empty(),
                )
                .sendable(),
            ),
        )
        .unwrap();
    }

    #[test]
    fn fails_on_top_level_nulls() {
        let strategy =
            StructStrategy::new(ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())));
        let res = block_on(
            strategy.write_stream(
                &ArrayContext::empty(),
                SequenceWriter::new(Box::new(TestSegments::default())),
                SequentialStreamAdapter::new(
                    DType::Struct(
                        Arc::new(
                            [("a", DType::Primitive(PType::I32, Nullability::NonNullable))]
                                .into_iter()
                                .collect(),
                        ),
                        Nullability::Nullable,
                    ),
                    stream::once(async move {
                        Ok((
                            SequenceId::root().downgrade(),
                            StructArray::try_new(
                                ["a".into()].into(),
                                vec![buffer![1, 2, 3].into_array()],
                                3,
                                Validity::Array(
                                    BoolArray::from_iter(vec![true, true, false]).into_array(),
                                ),
                            )
                            .unwrap()
                            .into_array(),
                        ))
                    }),
                )
                .sendable(),
            ),
        );
        assert!(
            format!("{}", res.unwrap_err())
                .starts_with("Cannot push struct chunks with top level invalid values"),
        )
    }
}
