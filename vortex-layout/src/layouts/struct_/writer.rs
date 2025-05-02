use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt as _;
use futures::future::try_join_all;
use itertools::Itertools;
use tokio::sync::Mutex;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::{Array, ArrayContext, ArrayRef, ToCanonical};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};

use crate::layouts::struct_::StructLayout;
use crate::scan::{TaskExecutor, TaskExecutorExt};
use crate::segments::ConcurrentSegmentWriter;
use crate::strategy::LayoutStrategy;
use crate::writer::LayoutWriter;
use crate::{IntoLayout, LayoutRef};

/// A [`LayoutWriter`] that splits a StructArray batch into child layout writers
pub struct StructLayoutWriter {
    column_strategies: Vec<Arc<Mutex<Box<dyn LayoutWriter>>>>,
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
                .map(|w| Arc::new(Mutex::new(w)))
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
