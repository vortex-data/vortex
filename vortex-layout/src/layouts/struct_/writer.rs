// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(deprecated, reason = "This module is deprecated")]

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::try_join_all;
use futures::pin_mut;
use itertools::Itertools;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::Handle;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_set::HashSet;

use crate::IntoLayout as _;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// A write strategy that shreds tabular data into columns and writes each column
/// as its own distinct stream.
///
/// This is now deprecated, users are encouraged to instead use the
/// [`TableStrategy`][crate::layouts::table::TableStrategy].
#[derive(Clone)]
#[deprecated(since = "0.59.0", note = "Use the `TableStrategy` instead.")]
pub struct StructStrategy {
    child: Arc<dyn LayoutStrategy>,
    validity: Arc<dyn LayoutStrategy>,
}

/// A [`LayoutStrategy`] that splits a StructArray batch into child layout writers
impl StructStrategy {
    pub fn new<S: LayoutStrategy, V: LayoutStrategy>(child: S, validity: V) -> Self {
        Self {
            child: Arc::new(child),
            validity: Arc::new(validity),
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
            return self
                .child
                .write_stream(ctx, segment_sink, stream, eof, handle)
                .await;
        };

        // Check for unique field names at write time.
        if HashSet::<_, DefaultHashBuilder>::from_iter(struct_dtype.names().iter()).len()
            != struct_dtype.names().len()
        {
            vortex_bail!("StructLayout must have unique field names");
        }

        let is_nullable = dtype.is_nullable();

        // Optimization: when there are no fields, don't spawn any work and just write a trivial
        // StructLayout.
        if struct_dtype.nfields() == 0 && !is_nullable {
            let row_count = stream
                .try_fold(
                    0u64,
                    |acc, (_, arr)| async move { Ok(acc + arr.len() as u64) },
                )
                .await?;
            return Ok(StructLayout::new(row_count, dtype, vec![]).into_layout());
        }

        // stream<struct_chunk> -> stream<vec<column_chunk>>
        let columns_vec_stream = stream.map(move |chunk| {
            let (sequence_id, chunk) = chunk?;
            let mut sequence_pointer = sequence_id.descend();
            let struct_chunk = chunk.to_struct();
            let mut columns: Vec<(SequenceId, ArrayRef)> = Vec::new();
            if is_nullable {
                columns.push((
                    sequence_pointer.advance(),
                    chunk.validity_mask()?.into_array(),
                ));
            }

            columns.extend(
                struct_chunk
                    .iter_unmasked_fields()
                    .map(|field| (sequence_pointer.advance(), field.to_array())),
            );

            Ok(columns)
        });

        let mut stream_count = struct_dtype.nfields();
        if is_nullable {
            stream_count += 1;
        }

        let (column_streams_tx, column_streams_rx): (Vec<_>, Vec<_>) =
            (0..stream_count).map(|_| kanal::bounded_async(1)).unzip();

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

        // First child column is the validity, subsequence children are the individual struct fields
        let column_dtypes: Vec<DType> = if is_nullable {
            std::iter::once(DType::Bool(Nullability::NonNullable))
                .chain(struct_dtype.fields())
                .collect()
        } else {
            struct_dtype.fields().collect()
        };

        let layout_futures: Vec<_> = column_dtypes
            .into_iter()
            .zip_eq(column_streams_rx)
            .enumerate()
            .map(move |(index, (dtype, recv))| {
                let column_stream =
                    SequentialStreamAdapter::new(dtype.clone(), recv.into_stream().boxed())
                        .sendable();
                let child_eof = eof.split_off();
                handle.spawn_nested(|h| {
                    let child = self.child.clone();
                    let validity = self.validity.clone();
                    let this = self.clone();
                    let ctx = ctx.clone();
                    let dtype = dtype.clone();
                    let segment_sink = segment_sink.clone();
                    async move {
                        // Write validity stream
                        if index == 0 && is_nullable {
                            validity
                                .write_stream(ctx, segment_sink, column_stream, child_eof, h)
                                .await
                        } else {
                            // Build recursive StructLayout for nested struct fields
                            // TODO(aduffy): add branch for ListLayout once that's implemented
                            if dtype.is_struct() {
                                this.write_stream(ctx, segment_sink, column_stream, child_eof, h)
                                    .await
                            } else {
                                child
                                    .write_stream(ctx, segment_sink, column_stream, child_eof, h)
                                    .await
                            }
                        }
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

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::Canonical;
    use vortex_array::IntoArray as _;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_io::runtime::single::block_on;

    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::struct_::writer::StructStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;

    #[test]
    #[should_panic]
    fn fails_on_duplicate_field() {
        let strategy =
            StructStrategy::new(FlatLayoutStrategy::default(), FlatLayoutStrategy::default());
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
    fn write_empty_field_struct_array() {
        let strategy =
            StructStrategy::new(FlatLayoutStrategy::default(), FlatLayoutStrategy::default());
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
