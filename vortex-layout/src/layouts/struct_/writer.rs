// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::channel::mpsc;
use futures::future::try_join_all;
use futures::{FutureExt, SinkExt, StreamExt, TryStreamExt, try_join};
use itertools::Itertools;
use vortex_array::{Array, ArrayContext, ToCanonical};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, vortex_bail};
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_set::HashSet;

use crate::layouts::struct_::StructLayout;
use crate::segments::SequenceWriter;
use crate::{
    IntoLayout as _, LayoutRef, LayoutStrategy, SendableSequentialStream, SequentialStreamAdapter,
    SequentialStreamExt, TaskExecutor, TaskExecutorExt,
};
pub struct StructStrategy<S> {
    child: S,
    executor: Arc<dyn TaskExecutor>,
    options: StructStrategyOptions,
}

/// A [`LayoutStrategy`] that splits a StructArray batch into child layout writers
impl<S> StructStrategy<S>
where
    S: LayoutStrategy,
{
    pub fn new(child: S, executor: Arc<dyn TaskExecutor>, options: StructStrategyOptions) -> Self {
        Self {
            child,
            executor,
            options,
        }
    }
}

pub struct StructStrategyOptions {
    buffer_size: usize,
}

impl Default for StructStrategyOptions {
    fn default() -> Self {
        Self { buffer_size: 512 }
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
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();

        let Some(struct_dtype) = stream.dtype().as_struct().cloned() else {
            return self.child.write_stream(ctx, sequence_writer, stream).await;
        };
        if HashSet::<_, DefaultHashBuilder>::from_iter(struct_dtype.names().iter()).len()
            != struct_dtype.names().len()
        {
            vortex_bail!("StructLayout must have unique field names");
        }

        let stream = stream.map(|chunk| {
            let (sequence_id, chunk) = chunk?;
            if !chunk.all_valid()? {
                vortex_bail!("Cannot push struct chunks with top level invalid values");
            };
            Ok((sequence_id, chunk))
        });

        // There are no fields so this is the layout leaf
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
        let mut columns_vec_stream = stream.map(|chunk| {
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

        let (mut column_txs, column_rxs): (Vec<_>, Vec<_>) = (0..struct_dtype.nfields())
            .map(|_| mpsc::channel(self.options.buffer_size))
            .unzip();

        let produce_handle = self.executor.spawn(
            async move {
                loop {
                    match columns_vec_stream.next().await {
                        Some(Ok(vec_columns)) => {
                            for (tx, chunk) in column_txs.iter_mut().zip_eq(vec_columns) {
                                let _ = tx.send(VortexResult::Ok(chunk)).await;
                            }
                        }
                        Some(Err(err)) => {
                            let shared: Arc<VortexError> = Arc::new(err);
                            for tx in column_txs.iter_mut() {
                                let _ = tx.send(Err(shared.clone().into())).await;
                            }
                        }
                        None => break,
                    }
                }
                Ok(())
            }
            .boxed(),
        );

        let column_dtypes = (0..struct_dtype.nfields()).map(move |idx| {
            struct_dtype
                .field_by_index(idx)
                .vortex_expect("bound checked")
        });

        let layout_futures: Vec<_> = column_dtypes
            .zip_eq(column_rxs)
            .map(move |(dtype, stream)| {
                let column_stream = SequentialStreamAdapter::new(dtype, stream).sendable();
                self.child
                    .write_stream(ctx, sequence_writer.clone(), column_stream)
                    .boxed()
            })
            .collect();

        let (column_layouts, _) = try_join!(try_join_all(layout_futures), produce_handle)?;
        // TODO(os): transposed stream could count row counts as well,
        // This must hold though, all columns must have the same row count of the struct layout
        let row_count = column_layouts.first().map(|l| l.row_count()).unwrap_or(0);
        Ok(StructLayout::new(row_count, dtype, column_layouts).into_layout())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use futures::stream;
    use vortex_array::arrays::{BoolArray, StructArray};
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayContext, IntoArray as _};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, FieldNames, Nullability, PType, StructFields};

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::struct_::writer::StructStrategy;
    use crate::segments::{SequenceWriter, TestSegments};
    use crate::sequence::SequenceId;
    use crate::{LayoutStrategy, LocalExecutor, SequentialStreamAdapter, SequentialStreamExt};

    #[test]
    #[should_panic]
    fn fails_on_duplicate_field() {
        let strategy = StructStrategy::new(
            FlatLayoutStrategy::default(),
            Arc::new(LocalExecutor),
            Default::default(),
        );
        block_on(
            strategy.write_stream(
                &ArrayContext::empty(),
                SequenceWriter::new(Box::new(TestSegments::default())),
                SequentialStreamAdapter::new(
                    DType::Struct(
                        [
                            ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                            ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                        ]
                        .into_iter()
                        .collect(),
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
        let strategy = StructStrategy::new(
            FlatLayoutStrategy::default(),
            Arc::new(LocalExecutor),
            Default::default(),
        );
        let res = block_on(
            strategy.write_stream(
                &ArrayContext::empty(),
                SequenceWriter::new(Box::new(TestSegments::default())),
                SequentialStreamAdapter::new(
                    DType::Struct(
                        [("a", DType::Primitive(PType::I32, Nullability::NonNullable))]
                            .into_iter()
                            .collect(),
                        Nullability::Nullable,
                    ),
                    stream::once(async move {
                        Ok((
                            SequenceId::root().downgrade(),
                            StructArray::try_new(
                                ["a"].into(),
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

    #[test]
    fn write_empty_field_struct_array() {
        let strategy = StructStrategy::new(
            FlatLayoutStrategy::default(),
            Arc::new(LocalExecutor),
            Default::default(),
        );
        let res = block_on(
            strategy.write_stream(
                &ArrayContext::empty(),
                SequenceWriter::new(Box::new(TestSegments::default())),
                SequentialStreamAdapter::new(
                    DType::Struct(
                        StructFields::new(FieldNames::default(), vec![]),
                        Nullability::NonNullable,
                    ),
                    stream::iter([
                        {
                            Ok((
                                SequenceId::root().downgrade(),
                                StructArray::try_new(
                                    FieldNames::default(),
                                    vec![],
                                    3,
                                    Validity::NonNullable,
                                )
                                .unwrap()
                                .into_array(),
                            ))
                        },
                        {
                            Ok((
                                SequenceId::root().advance(),
                                StructArray::try_new(
                                    FieldNames::default(),
                                    vec![],
                                    5,
                                    Validity::NonNullable,
                                )
                                .unwrap()
                                .into_array(),
                            ))
                        },
                    ]),
                )
                .sendable(),
            ),
        );

        assert_eq!(res.unwrap().row_count(), 8);
    }
}
