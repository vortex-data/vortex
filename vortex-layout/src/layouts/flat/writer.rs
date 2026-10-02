// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::serde::SerializeOptions;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::flat::flat_layout_inline_array_node;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;

/// Serialize one chunk and its cached finalized results into a flat layout.
#[derive(Clone)]
pub struct FlatLayoutStrategy {
    /// Whether to include padding for memory-mapped reads.
    pub include_padding: bool,
    /// Maximum length of serialized string and binary extrema in bytes.
    pub max_variable_length_statistics_size: usize,
}

impl Default for FlatLayoutStrategy {
    fn default() -> Self {
        Self {
            include_padding: true,
            max_variable_length_statistics_size: 64,
        }
    }
}

impl FlatLayoutStrategy {
    /// Set whether to include padding for memory-mapped reads.
    pub fn with_include_padding(mut self, include_padding: bool) -> Self {
        self.include_padding = include_padding;
        self
    }

    /// Set the maximum length of variable length statistics.
    pub fn with_max_variable_length_statistics_size(mut self, size: usize) -> Self {
        self.max_variable_length_statistics_size = size;
        self
    }
}

#[async_trait]
impl LayoutStrategy for FlatLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        _eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let Some(chunk) = stream.next().await else {
            // an empty input has no segment to write.
            return Ok(ChunkedLayout::new(
                0,
                stream.dtype().clone(),
                OwnedLayoutChildren::layout_children(vec![]),
            )
            .into_layout());
        };
        let (sequence_id, chunk) = chunk?;

        let row_count = chunk.len() as u64;

        let buffers = chunk.serialize(
            ctx.array_ctx(),
            session,
            &SerializeOptions {
                offset: 0,
                include_padding: self.include_padding,
                max_variable_length_statistics_size: Some(self.max_variable_length_statistics_size),
            },
        )?;
        // there is at least the flatbuffer and the length
        assert!(buffers.len() >= 2);
        let array_node =
            flat_layout_inline_array_node().then(|| buffers[buffers.len() - 2].clone());
        let segment_id = segment_sink.write(sequence_id, buffers).await?;

        let None = stream.next().await else {
            vortex_bail!("flat layout received stream with more than a single chunk");
        };
        Ok(FlatLayout::new_with_metadata(
            row_count,
            stream.dtype().clone(),
            segment_id,
            ReadContext::new(ctx.array_ctx().to_ids()),
            array_node,
        )
        .into_layout())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
    use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::min_max::min_max;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Precision;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBufferMut;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::AllOr;

    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;
    use crate::test::new_session;

    #[test]
    fn flat_layout_does_not_compute_results() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let reader = layout
                .new_reader("".into(), segments, &SESSION, &Default::default())
                .unwrap();
            let expr = root().bind(reader.dtype()).unwrap();
            let result = reader
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expr,
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            assert_eq!(
                result
                    .aggregations()
                    .get_result(&IsSorted.bind(IsSortedOptions { strict: false })),
                Precision::Absent
            );
        })
    }

    #[test]
    fn truncates_variable_size_stats() {
        block_on(|handle| async {
            let session = new_session().with_handle(handle);
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let mut builder = VarBinViewBuilder::with_capacity_in(
                DType::Utf8(Nullability::NonNullable),
                2,
                vortex_buffer::BufferAllocatorRef::statically_allocated(),
            );
            builder.append_value("Long value to test that the statistics are actually truncated, it needs a bit of extra padding though");
            builder.append_value("Another string that's meant to be smaller than the previous value, though still need extra padding");
            let array = builder.finish();
            let mut stats_ctx = session.create_execution_ctx();
            min_max(&array, &mut stats_ctx, NumericalAggregateOpts::skip_nans()).unwrap();
            let original_results = array.aggregations().snapshot_results();

            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx.into(),
                    Arc::<TestSegments>::clone(&segments),
                    array.clone().into_array().to_array_stream().sequenced(ptr),
                    eof,
                    &session,
                )
                .await
                .unwrap();

            let reader = layout
                .new_reader("".into(), segments, &SESSION, &Default::default())
                .unwrap();
            let expr = root().bind(reader.dtype()).unwrap();
            let result = reader
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expr,
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            for (aggregate, value) in original_results.iter() {
                assert_eq!(array.aggregations().get_result(aggregate), *value);
            }

            assert_eq!(
                result
                    .aggregations()
                    .get_result(&Min.bind(NumericalAggregateOpts::skip_nans())),
                // The typo is correct, we need this to be truncated.
                Precision::Inexact(Scalar::utf8(
                    // spellchecker:ignore-next-line
                    "Another string that's meant to be smaller than the previous valu",
                    Nullability::Nullable,
                ))
            );
            assert_eq!(
                result
                    .aggregations()
                    .get_result(&Max.bind(NumericalAggregateOpts::skip_nans())),
                Precision::Inexact(Scalar::utf8(
                    "Long value to test that the statistics are actually truncated, j",
                    Nullability::Nullable,
                ))
            );
        })
    }

    #[test]
    fn struct_array_round_trip() {
        block_on(|handle| async {
            let mut ctx_exec = array_session().create_execution_ctx();
            let session = new_session().with_handle(handle);
            let mut validity_builder = BitBufferMut::with_capacity(2);
            validity_builder.append(true);
            validity_builder.append(false);
            let validity_boolean_buffer = validity_builder.freeze();
            let validity = Validity::Array(
                BoolArray::new(validity_boolean_buffer.clone(), Validity::NonNullable).into_array(),
            );
            let array = StructArray::try_new(
                FieldNames::from([FieldName::from("a"), FieldName::from("b")]),
                vec![
                    buffer![1_u64, 2].into_array(),
                    buffer![3_u64, 4].into_array(),
                ],
                2,
                validity,
            )
            .unwrap();

            let ctx = ArrayContext::empty();

            // Write the array into a byte buffer.
            let (layout, segments) = {
                let segments = Arc::new(TestSegments::default());
                let (ptr, eof) = SequenceId::root().split();
                let layout = FlatLayoutStrategy::default()
                    .write_stream(
                        ctx.into(),
                        Arc::<TestSegments>::clone(&segments),
                        array.into_array().to_array_stream().sequenced(ptr),
                        eof,
                        &session,
                    )
                    .await
                    .unwrap();

                (layout, segments)
            };

            // We should be able to read the array we just wrote.
            let reader = layout
                .new_reader("".into(), segments, &SESSION, &Default::default())
                .unwrap();
            let expr = root().bind(reader.dtype()).unwrap();
            let result: ArrayRef = reader
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &expr,
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            assert_eq!(
                result
                    .validity()
                    .unwrap()
                    .execute_mask(result.len(), &mut ctx_exec)
                    .unwrap()
                    .bit_buffer(),
                AllOr::Some(&validity_boolean_buffer)
            );
            let result_struct = result
                .clone()
                .execute::<StructArray>(&mut ctx_exec)
                .unwrap();
            let field_a = result_struct
                .unmasked_field_by_name("a")
                .unwrap()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx_exec)
                .unwrap();
            assert_eq!(field_a.as_slice::<u64>(), &[1, 2]);
            let result_struct_b = result.execute::<StructArray>(&mut ctx_exec).unwrap();
            let field_b = result_struct_b
                .unmasked_field_by_name("b")
                .unwrap()
                .clone()
                .execute::<PrimitiveArray>(&mut ctx_exec)
                .unwrap();
            assert_eq!(field_b.as_slice::<u64>(), &[3, 4]);
        })
    }
}
