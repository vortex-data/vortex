// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListDataParts;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::fixed_size_list::FixedSizeListLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStream;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Strategy for writing fixed-size-list arrays, with a fallback for other dtypes.
///
/// Single-chunk only. For fixed-size-list input the strategy canonicalizes the chunk into a
/// [`FixedSizeListArray`] and writes the `elements` and optional `validity` columns into
/// independently configurable child strategies.
#[derive(Clone)]
pub struct FixedSizeListLayoutStrategy {
    elements: Arc<dyn LayoutStrategy>,
    validity: Arc<dyn LayoutStrategy>,
    fallback: Arc<dyn LayoutStrategy>,
}

impl Default for FixedSizeListLayoutStrategy {
    fn default() -> Self {
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        Self {
            elements: Arc::clone(&flat),
            validity: Arc::clone(&flat),
            fallback: flat,
        }
    }
}

impl FixedSizeListLayoutStrategy {
    /// Strategy for the `elements` child.
    pub fn with_elements(mut self, elements: Arc<dyn LayoutStrategy>) -> Self {
        self.elements = elements;
        self
    }

    /// Strategy for the `validity` child, written only when the list dtype is nullable.
    pub fn with_validity(mut self, validity: Arc<dyn LayoutStrategy>) -> Self {
        self.validity = validity;
        self
    }

    /// Strategy for non-fixed-size-list input, which is forwarded unchanged.
    pub fn with_fallback(mut self, fallback: Arc<dyn LayoutStrategy>) -> Self {
        self.fallback = fallback;
        self
    }
}

#[async_trait]
impl LayoutStrategy for FixedSizeListLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        if !dtype.is_fixed_size_list() {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        let Some(chunk) = stream.next().await else {
            vortex_bail!("FixedSizeListLayoutStrategy needs a single chunk");
        };
        let (sequence_id, array) = chunk?;

        let mut exec_ctx = session.create_execution_ctx();
        let row_count = array.len();
        let FixedSizeListDataParts {
            elements, validity, ..
        } = array
            .execute::<FixedSizeListArray>(&mut exec_ctx)?
            .into_data_parts();
        let validity_array = dtype
            .is_nullable()
            .then(|| {
                validity
                    .execute_mask(row_count, &mut exec_ctx)
                    .map(|m| m.into_array())
            })
            .transpose()?;

        let handle = session.handle();
        let (elements_task, validity_task) = {
            let mut sp = sequence_id.descend();
            let mut spawn_layout_writer = |strategy: Arc<dyn LayoutStrategy>, array: ArrayRef| {
                let stream = single_chunk_stream(array.dtype().clone(), sp.advance(), array);
                let child_eof = eof.split_off();
                let ctx = ctx.clone();
                let segment_sink = Arc::clone(&segment_sink);
                let session = session.clone();
                handle.spawn_nested(move |h| async move {
                    let session = session.with_handle(h);
                    strategy
                        .write_stream(ctx, segment_sink, stream, child_eof, &session)
                        .await
                })
            };
            (
                spawn_layout_writer(Arc::clone(&self.elements), elements),
                validity_array.map(|arr| spawn_layout_writer(Arc::clone(&self.validity), arr)),
            )
        };

        if stream.next().await.is_some() {
            vortex_bail!("FixedSizeListLayoutStrategy received more than a single chunk");
        }

        let (elements_layout, validity_layout) = futures::try_join!(elements_task, async move {
            match validity_task {
                Some(t) => t.await.map(Some),
                None => Ok(None),
            }
        },)?;

        Ok(
            FixedSizeListLayout::new(row_count as u64, dtype, elements_layout, validity_layout)
                .into_layout(),
        )
    }

    fn buffered_bytes(&self) -> u64 {
        let fsl_bytes = self.elements.buffered_bytes() + self.validity.buffered_bytes();
        fsl_bytes.max(self.fallback.buffered_bytes())
    }
}

fn single_chunk_stream(
    dtype: DType,
    sequence_id: SequenceId,
    array: ArrayRef,
) -> SendableSequentialStream {
    SequentialStreamAdapter::new(
        dtype,
        stream::once(async move { Ok((sequence_id, array)) }).boxed(),
    )
    .sendable()
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::*;
    use crate::segments::TestSegments;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    async fn write<S: LayoutStrategy>(strategy: &S, array: ArrayRef) -> VortexResult<LayoutRef> {
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        strategy
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await
    }

    fn create_fsl(validity: Validity) -> ArrayRef {
        FixedSizeListArray::new(buffer![1i32, 2, 3, 4, 5, 6].into_array(), 2, validity, 3)
            .into_array()
    }

    #[tokio::test]
    async fn basic_non_nullable_input() -> VortexResult<()> {
        let layout = write(
            &FixedSizeListLayoutStrategy::default(),
            create_fsl(Validity::NonNullable),
        )
        .await?;

        assert_eq!(
            layout.display_tree().to_string(),
            "vortex.fixed_size_list, dtype: fixed_size_list(i32)[2], children: 1\n\
             └── elements: vortex.flat, dtype: i32, segment: 0\n"
        );
        Ok(())
    }

    #[tokio::test]
    async fn basic_nullable_input() -> VortexResult<()> {
        let layout = write(
            &FixedSizeListLayoutStrategy::default(),
            create_fsl(Validity::Array(
                BoolArray::from_iter([true, false, true]).into_array(),
            )),
        )
        .await?;

        assert_eq!(
            layout.display_tree().to_string(),
            "vortex.fixed_size_list, dtype: fixed_size_list(i32)[2]?, children: 2\n\
             ├── elements: vortex.flat, dtype: i32, segment: 0\n\
             └── validity: vortex.flat, dtype: bool, segment: 1\n"
        );
        Ok(())
    }

    #[tokio::test]
    async fn non_fixed_size_list_input_routes_to_fallback() -> VortexResult<()> {
        let primitive = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let layout = write(&FixedSizeListLayoutStrategy::default(), primitive).await?;
        assert_eq!(
            layout.display_tree().to_string(),
            "vortex.flat, dtype: i32, segment: 0\n"
        );
        Ok(())
    }

    #[tokio::test]
    async fn empty_stream_errors() {
        let segments = Arc::new(TestSegments::default());
        let (_, eof) = SequenceId::root().split();
        let empty = stream::empty::<VortexResult<(SequenceId, ArrayRef)>>().boxed();
        let stream = SequentialStreamAdapter::new(
            DType::FixedSizeList(
                Arc::new(DType::Primitive(
                    vortex_array::dtype::PType::I32,
                    vortex_array::dtype::Nullability::NonNullable,
                )),
                2,
                vortex_array::dtype::Nullability::NonNullable,
            ),
            empty,
        )
        .sendable();

        let res = FixedSizeListLayoutStrategy::default()
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await;
        assert!(res.is_err());
    }
}
