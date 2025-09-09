// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;
use futures::{StreamExt, TryStreamExt, pin_mut};
use itertools::Itertools;
use vortex_array::{Array, ArrayContext, ToCanonical};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, vortex_bail};
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::Handle;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_set::HashSet;

use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::{
    SendableSequentialStream, SequencePointer, SequentialStreamAdapter, SequentialStreamExt,
};
use crate::{IntoLayout as _, LayoutRef, LayoutStrategy};

pub struct StructStrategy {
    child: Arc<dyn LayoutStrategy>,
}

/// A [`LayoutStrategy`] that splits a StructArray batch into child layout writers
impl StructStrategy {
    pub fn new<S: LayoutStrategy>(child: S) -> Self {
        Self {
            child: Arc::new(child),
        }
    }
}

#[async_trait]
impl LayoutStrategy for StructStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let Some(struct_dtype) = stream.dtype().as_struct_fields_opt().cloned() else {
            // nothing we can do if dtype is not struct
            return self
                .child
                .write_stream(ctx, segment_sink, stream, eof, handle)
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

        let (column_streams_tx, column_streams_rx): (Vec<_>, Vec<_>) = (0..struct_dtype.nfields())
            .map(|_| kanal::unbounded_async())
            .unzip();

        // Spawn a task to fan out column chunks to their respective transposed streams
        handle
            .spawn(async move {
                pin_mut!(columns_vec_stream);
                while let Some(result) = columns_vec_stream.next().await {
                    match result {
                        Ok(columns) => {
                            for (tx, column) in column_streams_tx.iter().zip_eq(columns.into_iter())
                            {
                                let _ = tx.send(Ok(column)).await;
                            }
                        }
                        Err(e) => {
                            let e: Arc<VortexError> = Arc::new(e);
                            for tx in column_streams_tx.iter() {
                                let _ = tx.send(Err(VortexError::from(e.clone()))).await;
                            }
                            break;
                        }
                    }
                }
            })
            .detach();

        let column_dtypes = (0..struct_dtype.nfields()).map(move |idx| {
            struct_dtype
                .field_by_index(idx)
                .vortex_expect("bound checked")
        });

        let layout_futures: Vec<_> = column_dtypes
            .zip_eq(column_streams_rx)
            .map(move |(dtype, recv)| {
                let column_stream =
                    SequentialStreamAdapter::new(dtype, recv.into_stream().boxed()).sendable();
                let child_eof = eof.split_off();
                handle.spawn_nested(|h| {
                    let child = self.child.clone();
                    let ctx = ctx.clone();
                    let segment_sink = segment_sink.clone();
                    async move {
                        child
                            .write_stream(ctx, segment_sink, column_stream, child_eof, h)
                            .await
                    }
                })
            })
            .collect();

        let column_layouts = try_join_all(layout_futures).await?;
        // TODO(os): transposed stream could count row counts as well,
        // This must hold though, all columns must have the same row count of the struct layout
        let row_count = column_layouts.first().map(|l| l.row_count()).unwrap_or(0);
        Ok(StructLayout::new(row_count, dtype, column_layouts).into_layout())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::arrays::{BoolArray, ChunkedArray, StructArray};
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayContext, Canonical, IntoArray as _};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, FieldNames, Nullability, PType};
    use vortex_io::runtime::single::SingleThreadRuntime;

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
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        SingleThreadRuntime::block_on(|handle| {
            strategy.write_stream(
                ctx,
                segments,
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
                handle,
            )
        })
        .unwrap();
    }

    #[test]
    fn fails_on_top_level_nulls() {
        let strategy = StructStrategy::new(FlatLayoutStrategy::default());
        let (ptr, eof) = SequenceId::root().split();
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let res = SingleThreadRuntime::block_on(|handle| {
            strategy.write_stream(
                ctx,
                segments,
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
                handle,
            )
        });
        assert!(
            format!("{}", res.unwrap_err())
                .starts_with("Cannot push struct chunks with top level invalid values"),
        )
    }

    #[test]
    fn write_empty_field_struct_array() {
        let strategy = StructStrategy::new(FlatLayoutStrategy::default());
        let (ptr, eof) = SequenceId::root().split();
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let res = SingleThreadRuntime::block_on(|handle| {
            strategy.write_stream(
                ctx,
                segments,
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
                handle,
            )
        });

        assert_eq!(res.unwrap().row_count(), 8);
    }
}
