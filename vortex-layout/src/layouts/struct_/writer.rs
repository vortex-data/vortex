use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use async_trait::async_trait;
use futures::future::try_join_all;
use futures::{FutureExt as _, Stream, StreamExt as _};
use itertools::Itertools;
use parking_lot::Mutex;
use tokio::sync::Mutex as TokioMutex;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::arcref::ArcRef;
use vortex_array::{Array, ArrayContext, ArrayRef, ToCanonical};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};

use crate::layouts::struct_::StructLayout;
use crate::scan::{TaskExecutor, TaskExecutorExt};
use crate::segments::{ConcurrentSegmentWriter, NewSegmentWriter};
use crate::strategy::LayoutStrategy;
use crate::writer::LayoutWriter;
use crate::{LayoutVTableRef, NewLayoutStrategy, NewLayoutWriter, SequentialArrayStream};

pub struct NewStructStrategy {
    child: ArcRef<dyn NewLayoutStrategy>,
}

impl NewLayoutStrategy for NewStructStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        dtype: &DType,
        segment_writer: Arc<dyn NewSegmentWriter>,
        stream: SequentialArrayStream,
    ) -> Pin<Box<dyn NewLayoutWriter>> {
        let Some(struct_dtype) = dtype.as_struct().cloned() else {
            return Box::pin(async { Err(vortex_err!("expected StructDType")) });
        };
        if HashSet::from_iter(struct_dtype.names().iter()).len() != struct_dtype.names().len() {
            return Box::pin(async {
                Err(vortex_err!("StructLayout must have unique field names"))
            });
        }

        // stream<struct_chunk> -> stream<vec<column_chunk>>
        let columns_vec_stream = stream.map(|chunk| {
            let (sequence_id, chunk) = chunk?;
            let (_, mut sequence_pointer) = sequence_id.descend();
            let struct_chunk = chunk
                .as_struct_typed()
                .ok_or_else(|| vortex_err!("chunk is not struct typed"))?;
            let columns: Vec<_> = (0..struct_chunk.nfields())
                .map(|idx| {
                    (
                        sequence_pointer.advance(),
                        struct_chunk
                            .maybe_null_field_by_idx(idx)
                            .vortex_expect("bounds already checked"),
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
        let layout_futures =
            column_dtypes
                .zip_eq(column_streams.into_iter())
                .map(move |(dtype, stream)| {
                    child.write_stream(&ctx, &dtype, segment_writer.clone(), Box::pin(stream))
                });

        let dtype = dtype.clone();
        Box::pin(async move {
            let column_layouts = try_join_all(layout_futures).await?;
            // TODO(os): transposed stream could count row counts as well,
            // This must hold though, all columns must have the same row count of the struct layout
            let row_count = column_layouts.get(0).map(|l| l.row_count()).unwrap_or(0);
            Ok(Layout::new_owned(
                "struct".into(),
                LayoutVTableRef::new_ref(&StructLayout),
                dtype,
                row_count,
                vec![],
                column_layouts,
                None,
            ))
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
                Poll::Ready(Some(Err(shared_err.clone().into())))
            }
        }
    }
}
use crate::{IntoLayout, LayoutRef};

/// A [`LayoutWriter`] that splits a StructArray batch into child layout writers
pub struct StructLayoutWriter {
    column_strategies: Vec<Arc<TokioMutex<Box<dyn LayoutWriter>>>>,
    dtype: DType,
    executor: Option<Arc<dyn TaskExecutor>>,
    row_count: u64,
}

impl StructLayoutWriter {
    pub fn try_new(
        dtype: DType,
        executor: Option<Arc<dyn TaskExecutor>>,
        column_layout_writers: Vec<Box<dyn LayoutWriter>>,
    ) -> VortexResult<Self> {
        let struct_dtype = dtype
            .as_struct()
            .ok_or_else(|| vortex_err!("expected StructDType"))?;
        if HashSet::from_iter(struct_dtype.names().iter()).len() != struct_dtype.names().len() {
            vortex_bail!("StructLayout must have unique field names")
        }
        if struct_dtype.fields().len() != column_layout_writers.len() {
            vortex_bail!(
                "number of fields in struct dtype does not match number of column layout writers"
            );
        }
        Ok(Self {
            column_strategies: column_layout_writers
                .into_iter()
                .map(|w| Arc::new(TokioMutex::new(w)))
                .collect(),
            dtype,
            executor,
            row_count: 0,
        })
    }

    pub fn try_new_with_strategy<S: LayoutStrategy>(
        ctx: &ArrayContext,
        dtype: &DType,
        executor: Option<Arc<dyn TaskExecutor>>,
        factory: &S,
    ) -> VortexResult<Self> {
        let struct_dtype = dtype
            .as_struct()
            .ok_or_else(|| vortex_err!("expected StructDType"))?;
        Self::try_new(
            dtype.clone(),
            executor,
            struct_dtype
                .fields()
                .map(|field_dtype| factory.new_writer(ctx, &field_dtype))
                .try_collect()?,
        )
    }
}

#[async_trait]
impl LayoutWriter for StructLayoutWriter {
    async fn push_chunk(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        assert_eq!(
            chunk.dtype(),
            &self.dtype,
            "Can't push chunks of the wrong dtype into a LayoutWriter. Pushed {} but expected {}.",
            chunk.dtype(),
            self.dtype
        );
        let struct_array = chunk.to_struct()?;
        if struct_array.struct_dtype().nfields() != self.column_strategies.len() {
            vortex_bail!(
                "number of fields in struct array does not match number of column layout writers"
            );
        }
        self.row_count += struct_array.len() as u64;

        let column_futures = segment_writer
            .split_off(struct_array.nfields())?
            .into_iter()
            .enumerate()
            .map(|(i, mut writer)| {
                // TODO(joe): handle struct validity
                let column = chunk
                    .as_struct_typed()
                    .vortex_expect("batch is a struct array")
                    .maybe_null_field_by_idx(i)
                    .vortex_expect("bounds already checked");
                let col_strategy = self.column_strategies[i].clone();
                let column_fut = async move {
                    for column_chunk in column.to_array_iterator() {
                        col_strategy
                            .lock()
                            .await
                            .push_chunk(&mut *writer, column_chunk?)
                            .await?;
                    }
                    Ok(())
                }
                .boxed();
                match &self.executor {
                    Some(exec) => exec.spawn(column_fut),
                    None => column_fut,
                }
            })
            .collect_vec();
        try_join_all(column_futures).await?;
        Ok(())
    }

    async fn flush(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
    ) -> VortexResult<()> {
        for writer in self.column_strategies.iter_mut() {
            writer.lock().await.flush(segment_writer).await?;
        }
        Ok(())
    }

    async fn finish(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
    ) -> VortexResult<LayoutRef> {
        let mut column_layouts = vec![];
        for writer in self.column_strategies.iter_mut() {
            column_layouts.push(writer.lock().await.finish(segment_writer).await?);
        }
        Ok(StructLayout::new(self.row_count, self.dtype.clone(), column_layouts).into_layout())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::LayoutWriterExt;
    use crate::layouts::flat::writer::{FlatLayoutStrategy, FlatLayoutWriter};
    use crate::layouts::struct_::writer::StructLayoutWriter;

    #[test]
    #[should_panic]
    fn fails_on_duplicate_field() {
        StructLayoutWriter::try_new(
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
            vec![
                FlatLayoutWriter::new(
                    ArrayContext::empty(),
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    FlatLayoutStrategy::default(),
                )
                .boxed(),
                FlatLayoutWriter::new(
                    ArrayContext::empty(),
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    FlatLayoutStrategy::default(),
                )
                .boxed(),
            ],
        )
        .unwrap();
    }
}
