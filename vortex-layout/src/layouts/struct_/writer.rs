// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;
use futures::{StreamExt, TryStreamExt, pin_mut};
use itertools::Itertools;
use vortex_array::arrays::StructArray;
use vortex_array::{Array, ArrayContext, ArrayRef, ToCanonical};
use vortex_dtype::{DType, FieldName, FieldNames, Nullability, StructFields};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, vortex_bail};
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::Handle;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashMap;
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

        // TODO(aduffy): stitch nested fields all back together

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

        fn extract_field(chunk: &StructArray, field_path: &[usize]) -> ArrayRef {
            match *field_path {
                [] => chunk.to_array(),
                [idx] => chunk.fields()[idx].clone(),
                [idx, ref rest @ ..] => {
                    let parent = chunk.fields()[idx].clone();
                    extract_field(&parent.to_struct(), rest)
                }
            }
        }

        fn assign_field_paths(
            fields: &StructFields,
            field_path: Vec<usize>,
            field_name: Vec<String>,
            field_paths: &mut Vec<Vec<usize>>,
            field_types: &mut HashMap<Vec<usize>, DType>,
            field_names: &mut HashMap<Vec<usize>, FieldName>,
        ) {
            for (index, dtype) in fields.fields().enumerate() {
                let name = fields.field_name(index).vortex_expect("field name");
                let mut new_field = field_path.clone();
                let mut new_name = field_name.clone();
                new_field.push(index);
                new_name.push(name.to_string());

                if let Some(struct_field) = dtype.as_struct_fields_opt() {
                    // recursively assign field paths
                    assign_field_paths(
                        struct_field,
                        new_field,
                        new_name,
                        field_paths,
                        field_types,
                        field_names,
                    );
                } else {
                    field_types.insert(new_field.clone(), dtype);
                    field_names.insert(
                        new_field.clone(),
                        FieldName::from(new_name.iter().join(".")),
                    );
                    field_paths.push(new_field);
                }
            }
        }

        let mut field_paths = Vec::with_capacity(struct_dtype.nfields());
        let mut field_dtypes = HashMap::new();
        let mut field_names = HashMap::new();
        assign_field_paths(
            &struct_dtype,
            vec![],
            vec![],
            &mut field_paths,
            &mut field_dtypes,
            &mut field_names,
        );

        let field_paths2 = field_paths.clone();

        // stream<struct_chunk> -> stream<vec<column_chunk>>
        let columns_vec_stream = stream.map(move |chunk| {
            let (sequence_id, chunk) = chunk?;
            let mut sequence_pointer = sequence_id.descend();
            let struct_chunk = chunk.to_struct();
            let columns: Vec<_> = field_paths2
                .iter()
                .map(|field_path| {
                    (
                        sequence_pointer.advance(),
                        extract_field(&struct_chunk, field_path),
                    )
                })
                .collect();

            Ok(columns)
        });

        let (column_streams_tx, column_streams_rx): (Vec<_>, Vec<_>) = (0..field_paths.len())
            .map(|_| kanal::bounded_async(1))
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

        let column_dtypes = field_paths
            .iter()
            .map(|path| field_dtypes.get(path).vortex_expect("must work").clone());

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

        let flat_names = field_paths
            .iter()
            .map(|path| field_names.get(path).vortex_expect("must work").clone())
            .collect_vec();
        let flat_dtypes = field_paths
            .iter()
            .map(|path| field_dtypes.get(path).vortex_expect("must work").clone())
            .collect_vec();
        let flattened_struct_dtype = StructFields::new(FieldNames::from(flat_names), flat_dtypes);

        Ok(StructLayout::new(
            row_count,
            DType::Struct(flattened_struct_dtype, dtype.nullability()),
            column_layouts,
        )
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes()
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
    use vortex_io::runtime::single::block_on;

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
        block_on(|handle| {
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
        let res = block_on(|handle| {
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
        let res = block_on(|handle| {
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
