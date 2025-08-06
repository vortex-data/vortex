//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::once;
use vortex_array::arrays::VarBinViewVTable;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_array::{Array, ArrayContext};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_bail};

use crate::children::OwnedLayoutChildren;
use crate::layouts::view::{ValidityTag, ViewLayout};
use crate::segments::SequenceWriter;
use crate::{
    IntoLayout, LayoutRef, LayoutStrategy, SendableSequentialStream, SequentialStreamAdapter,
    SequentialStreamExt,
};

/// Strategy for writing a VarBinView arrays with multiple buffers.
///
/// This will yield `ViewLayout`s, which at scan time can eliminate many buffer reads that
/// are unnecessary, improving performance for arrays with large values.
#[derive(Clone)]
pub struct ViewStrategy<ValidityStrategy, FallbackStrategy> {
    pub(crate) validity_strategy: ValidityStrategy,
    pub(crate) fallback_strategy: FallbackStrategy,
}

impl<ValidityStrategy, FallbackStrategy> ViewStrategy<ValidityStrategy, FallbackStrategy>
where
    ValidityStrategy: LayoutStrategy,
    FallbackStrategy: LayoutStrategy,
{
    pub fn new(validity_strategy: ValidityStrategy, fallback_strategy: FallbackStrategy) -> Self {
        Self {
            validity_strategy,
            fallback_strategy,
        }
    }
}

const VALIDITY_DTYPE: DType = DType::Bool(Nullability::NonNullable);

#[async_trait]
impl<ValidityStrategy, FallbackStrategy> LayoutStrategy
    for ViewStrategy<ValidityStrategy, FallbackStrategy>
where
    ValidityStrategy: LayoutStrategy,
    FallbackStrategy: LayoutStrategy,
{
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        writer: SequenceWriter,
        mut stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
        let ctx = ctx.clone();

        let Some(chunk) = stream.next().await else {
            vortex_bail!("ViewLayout needs a single chunk");
        };
        let (sequence_id, chunk) = chunk?;
        let row_count = chunk.len() as u64;

        // If the chunk is a VarBinView, serialize using our specialized layout.
        if let Some(view_array) = chunk.as_opt::<VarBinViewVTable>() {
            let mut ptr = sequence_id.descend();

            let view_id = ptr.advance();

            let views = view_array.views().clone().into_byte_buffer();
            let views_segment = writer.put(view_id, vec![views]).await?;

            let mut buffer_segments = Vec::new();
            for buffer in view_array.buffers().iter() {
                let buf_id = ptr.advance();
                let buf_segment = writer.put(buf_id, vec![buffer.clone()]).await?;
                buffer_segments.push(buf_segment);
            }

            // Write the validity child, if present.
            // Record metadata about if the validity is all valid, all invalid, or an array.
            let validity_tag = match view_array.validity() {
                Validity::NonNullable => ValidityTag::NonNullable,
                Validity::AllValid => ValidityTag::AllValid,
                Validity::AllInvalid => ValidityTag::AllInvalid,
                Validity::Array(_) => ValidityTag::Array,
            };

            let children = if let Some(validity_array) = view_array.validity().clone().into_array()
            {
                let child = self
                    .validity_strategy
                    .write_stream(
                        &ctx,
                        writer,
                        SequentialStreamAdapter::new(
                            VALIDITY_DTYPE,
                            once(async move { Ok((ptr.advance(), validity_array)) }),
                        )
                        .sendable(),
                    )
                    .await?;

                vec![child]
            } else {
                vec![]
            };

            Ok(ViewLayout::new(
                row_count,
                chunk.dtype().clone(),
                validity_tag,
                views_segment,
                buffer_segments,
                OwnedLayoutChildren::layout_children(children),
                ctx.clone(),
            )
            .into_layout())
        } else {
            self.fallback_strategy
                .write_stream(&ctx, writer, stream)
                .await
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arcref::ArcRef;
    use futures::executor::block_on;
    use futures::stream::once;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::{ArrayContext, IntoArray};
    use vortex_dtype::{DType, Nullability};

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::view::ViewVTable;
    use crate::layouts::view::writer::ViewStrategy;
    use crate::segments::{SequenceWriter, TestSegments};
    use crate::sequence::SequenceId;
    use crate::{LayoutChildren, LayoutStrategy, SequentialStreamAdapter, SequentialStreamExt};

    #[test]
    fn test_write_view() {
        // Write a new ViewLayout from an input stream of chunks.
        let ctx = ArrayContext::empty();
        let strategy = ViewStrategy {
            validity_strategy: ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())),
            fallback_strategy: ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())),
        };

        let writer = Box::new(TestSegments::default());
        let writer = SequenceWriter::new(writer);
        let mut sequence_id = SequenceId::root();

        let stream = SequentialStreamAdapter::new(
            DType::Utf8(Nullability::NonNullable),
            once(async move {
                let array = VarBinViewArray::from_iter_str([
                    "inlined1",
                    "inlined2",
                    "this string will be outlined",
                ])
                .into_array();

                Ok((sequence_id.advance(), array))
            }),
        )
        .sendable();

        let written = block_on(strategy.write_stream(&ctx, writer, stream)).unwrap();
        assert!(written.is::<ViewVTable>());
        let view_layout = written.as_::<ViewVTable>();
        assert_eq!(view_layout.children.nchildren(), 0);
        assert_eq!(view_layout.buffers.len(), 1);
        assert_eq!(view_layout.row_count, 3);
    }
}
