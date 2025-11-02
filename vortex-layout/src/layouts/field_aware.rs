// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use async_trait::async_trait;
use futures::future::try_join_all;
use futures::{StreamExt, TryStreamExt, pin_mut, stream};
use kanal::bounded_async;
use vortex_array::{ArrayContext, ToCanonical};
use vortex_dtype::FieldName;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail};
use vortex_io::runtime::Handle;

use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::{
    SendableSequentialStream, SequencePointer, SequentialStreamAdapter, SequentialStreamExt,
};
use crate::{IntoLayout, LayoutRef, LayoutStrategy};

/// Strategy that delegates writing of struct fields to field-specific layout strategies.
///
/// A default strategy is used for every field unless an override is registered for that
/// field name. This mirrors [`StructStrategy`] in the upstream Vortex writer while enabling
/// per-column customisation (e.g. attaching a bloom filter layout for selected UTF-8 columns).
pub struct FieldAwareStructStrategy {
    default: Arc<dyn LayoutStrategy>,
    overrides: HashMap<FieldName, Arc<dyn LayoutStrategy>>,
}

impl FieldAwareStructStrategy {
    /// Create a new [`FieldAwareStructStrategy`] with the provided default child strategy.
    pub fn new<S: LayoutStrategy>(default: S) -> Self {
        Self {
            default: Arc::new(default),
            overrides: HashMap::new(),
        }
    }

    /// Register a strategy override for a single field name.
    pub fn with_override<S, N>(mut self, field: N, strategy: S) -> Self
    where
        S: LayoutStrategy,
        N: Into<FieldName>,
    {
        self.overrides.insert(field.into(), Arc::new(strategy));
        self
    }

    fn strategy_for_field(&self, field: &FieldName) -> Arc<dyn LayoutStrategy> {
        self.overrides
            .get(field)
            .cloned()
            .unwrap_or_else(|| self.default.clone())
    }
}

#[async_trait]
impl LayoutStrategy for FieldAwareStructStrategy {
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
            // Nothing special to do if the input is not a struct: delegate to the default
            // layout strategy exactly as Vortex would have done before introducing this wrapper.
            return self
                .default
                .write_stream(ctx, segment_sink, stream, eof, handle)
                .await;
        };

        let mut seen = HashSet::with_capacity(struct_dtype.nfields());
        for field in struct_dtype.names().iter().cloned() {
            if !seen.insert(field) {
                vortex_bail!("FieldAwareStructStrategy requires unique field names");
            }
        }

        let stream = stream.map(|chunk| {
            let (sequence_id, chunk) = chunk?;
            if !chunk.all_valid() {
                vortex_bail!("Cannot push struct chunks with top level invalid values");
            }
            Ok((sequence_id, chunk))
        });

        if struct_dtype.nfields() == 0 {
            // Zero-field structs only contribute a row count, so we can short-circuit here.
            let row_count = stream
                .try_fold(
                    0u64,
                    |acc, (_, arr)| async move { Ok(acc + arr.len() as u64) },
                )
                .await?;
            return Ok(StructLayout::new(row_count, dtype, vec![]).into_layout());
        }

        // stream<struct_chunk> -> stream<Vec<(SequencePointer, ArrayRef)>>, one entry per column.
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
            VortexResult::Ok(columns)
        });

        // Create a dedicated async channel per column so each one can be written concurrently.
        let (senders, receivers): (Vec<_>, Vec<_>) = (0..struct_dtype.nfields())
            .map(|_| bounded_async(1))
            .unzip();

        handle
            .spawn(async move {
                pin_mut!(columns_vec_stream);
                while let Some(result) = columns_vec_stream.next().await {
                    match result {
                        Ok(columns) => {
                            for (sender, column) in senders.iter().zip(columns.into_iter()) {
                                if sender.send(Ok(column)).await.is_err() {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let err: Arc<VortexError> = Arc::new(e);
                            for sender in &senders {
                                let _ = sender.send(Err(VortexError::from(err.clone()))).await;
                            }
                            break;
                        }
                    }
                }
            })
            .detach();

        let layout_futures = receivers
            .into_iter()
            .enumerate()
            .map(|(idx, receiver)| {
                let dtype = struct_dtype
                    .field_by_index(idx)
                    .vortex_expect("field index should exist");
                let field_name = struct_dtype
                    .field_name(idx)
                    .cloned()
                    .vortex_expect("field name should exist");
                let strategy = self.strategy_for_field(&field_name);
                let column_stream = SequentialStreamAdapter::new(
                    dtype,
                    stream::unfold(receiver, |receiver| async move {
                        match receiver.recv().await {
                            Ok(value) => Some((value, receiver)),
                            Err(_) => None,
                        }
                    }),
                )
                .sendable();
                let child_eof = eof.split_off();
                let ctx = ctx.clone();
                let segment_sink = segment_sink.clone();
                handle.spawn_nested(move |h| async move {
                    strategy
                        .write_stream(ctx, segment_sink, column_stream, child_eof, h)
                        .await
                })
            })
            .collect::<Vec<_>>();

        let column_layouts = try_join_all(layout_futures).await?;
        let row_count = column_layouts
            .first()
            .map(|layout| layout.row_count())
            .unwrap_or(0);

        Ok(StructLayout::new(row_count, dtype, column_layouts).into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.overrides
            .values()
            .fold(self.default.buffered_bytes(), |acc, strategy| {
                acc.max(strategy.buffered_bytes())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_array::IntoArray;
    use vortex_array::arrays::{StructArray, VarBinArray};
    use vortex_dtype::{DType, Nullability};
    use vortex_io::runtime::single::block_on;

    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::FlatVTable;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::struct_::StructVTable;
    use crate::layouts::zoned::ZonedVTable;
    use crate::layouts::zoned::writer::{ZonedLayoutOptions, ZonedStrategy};
    use crate::segments::TestSegments;
    use crate::sequence::{SequenceId, SequentialArrayStreamExt};

    #[test]
    fn applies_field_overrides() {
        let array = StructArray::from_fields(&[
            (
                "a",
                VarBinArray::from_iter(
                    ["alpha", "beta"].into_iter().map(Some),
                    DType::Utf8(Nullability::NonNullable),
                )
                .into_array(),
            ),
            (
                "b",
                VarBinArray::from_iter(
                    ["gamma", "delta"].into_iter().map(Some),
                    DType::Utf8(Nullability::NonNullable),
                )
                .into_array(),
            ),
        ])
        .unwrap()
        .into_array();
        let (ptr, eof) = SequenceId::root().split();
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());

        let default = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default());
        let chunked_override = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions::default(),
        );

        let strategy = FieldAwareStructStrategy::new(default).with_override("a", chunked_override);

        let layout = block_on(|handle| async {
            strategy
                .write_stream(
                    ctx,
                    segments,
                    array.to_array_stream().sequenced(ptr),
                    eof,
                    handle,
                )
                .await
        })
        .unwrap();

        let struct_layout = layout.as_::<StructVTable>();
        let a_layout = struct_layout.child(0).unwrap();
        let b_layout = struct_layout.child(1).unwrap();

        assert!(a_layout.is::<ZonedVTable>());
        // not enough rows to warrant chunks
        assert!(b_layout.is::<FlatVTable>());
    }
}
