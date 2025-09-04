// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::serde::SerializeOptions;
use vortex_array::stats::{Precision, Stat, StatsProvider};
use vortex_array::{Array, ArrayContext};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{BinaryScalar, Utf8Scalar};

use crate::layouts::flat::FlatLayout;
use crate::layouts::zoned::{lower_bound, upper_bound};
use crate::segments::SequenceWriter;
use crate::{IntoLayout, LayoutRef, LayoutStrategy, SendableSequentialStream};

#[derive(Clone)]
pub struct FlatLayoutStrategy {
    /// Whether to include padding for memory-mapped reads.
    pub include_padding: bool,
    /// Maximum length of variable length statistics
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

#[async_trait]
impl LayoutStrategy for FlatLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        mut stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
        let ctx = ctx.clone();
        let options = self.clone();
        let Some(chunk) = stream.next().await else {
            vortex_bail!("flat layout needs a single chunk");
        };
        let (sequence_id, chunk) = chunk?;

        let row_count = chunk.len() as u64;

        match chunk.dtype() {
            DType::Utf8(_) => {
                if let Some(sv) = chunk.statistics().get(Stat::Min) {
                    let (value, truncated) = lower_bound::<Utf8Scalar>(
                        sv.into_inner(),
                        options.max_variable_length_statistics_size,
                    )?;
                    if truncated {
                        chunk
                            .statistics()
                            .set(Stat::Min, Precision::Inexact(value.into_value()));
                    }
                }

                if let Some(sv) = chunk.statistics().get(Stat::Max) {
                    let (value, truncated) = upper_bound::<Utf8Scalar>(
                        sv.into_inner(),
                        options.max_variable_length_statistics_size,
                    )?;
                    if let Some(upper_bound) = value {
                        if truncated {
                            chunk
                                .statistics()
                                .set(Stat::Max, Precision::Inexact(upper_bound.into_value()));
                        }
                    } else {
                        chunk.statistics().clear(Stat::Max)
                    }
                }
            }
            DType::Binary(_) => {
                if let Some(sv) = chunk.statistics().get(Stat::Min) {
                    let (value, truncated) = lower_bound::<BinaryScalar>(
                        sv.into_inner(),
                        options.max_variable_length_statistics_size,
                    )?;
                    if truncated {
                        chunk
                            .statistics()
                            .set(Stat::Min, Precision::Inexact(value.into_value()));
                    }
                }

                if let Some(sv) = chunk.statistics().get(Stat::Max) {
                    let (value, truncated) = upper_bound::<BinaryScalar>(
                        sv.into_inner(),
                        options.max_variable_length_statistics_size,
                    )?;
                    if let Some(upper_bound) = value {
                        if truncated {
                            chunk
                                .statistics()
                                .set(Stat::Max, Precision::Inexact(upper_bound.into_value()));
                        }
                    } else {
                        chunk.statistics().clear(Stat::Max)
                    }
                }
            }
            _ => {}
        }

        // TODO(os): spawn serialization
        let buffers = chunk.serialize(
            &ctx,
            &SerializeOptions {
                offset: 0,
                include_padding: options.include_padding,
            },
        )?;
        let segment_id = sequence_writer.put(sequence_id, buffers).await?;

        let None = stream.next().await else {
            vortex_bail!("flat layout received stream with more than a single chunk");
        };
        Ok(
            FlatLayout::new(row_count, stream.dtype().clone(), segment_id, ctx.clone())
                .into_layout(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_buffer::BooleanBufferBuilder;
    use futures::executor::block_on;
    use futures::stream;
    use vortex_array::arrays::{BoolArray, PrimitiveArray, StructArray};
    use vortex_array::builders::{ArrayBuilder, VarBinViewBuilder};
    use vortex_array::pipeline::operators::MaskFuture;
    use vortex_array::stats::{Precision, Stat, StatsProviderExt};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayContext, ArrayRef, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, FieldName, FieldNames, Nullability};
    use vortex_error::VortexUnwrap;
    use vortex_expr::root;
    use vortex_mask::AllOr;

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::{SegmentSource, SequenceWriter, TestSegments};
    use crate::sequence::SequenceId;
    use crate::{
        LayoutStrategy, SendableSequentialStream, SequentialStreamAdapter, SequentialStreamExt as _,
    };

    fn stream_only(array: ArrayRef) -> SendableSequentialStream {
        SequentialStreamAdapter::new(
            array.dtype().clone(),
            stream::once(async move { Ok((SequenceId::root().downgrade(), array)) }),
        )
        .sendable()
    }

    // Currently, flat layouts do not force compute stats during write, they only retain
    // pre-computed stats.
    #[should_panic]
    #[test]
    fn flat_stats() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let segments = TestSegments::default();
            let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutStrategy::default()
                .write_stream(&ctx, sequence_writer, stream_only(array.to_array()))
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let result = layout
                .new_reader("".into(), segments)
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &root(),
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            assert_eq!(
                result.statistics().get_as::<bool>(Stat::IsSorted),
                Some(Precision::Exact(true))
            );
        })
    }

    #[test]
    fn truncates_variable_size_stats() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let segments = TestSegments::default();
            let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
            let mut builder =
                VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::NonNullable), 2);
            builder.append_value("Long value to test that the statistics are actually truncated, it needs a bit of extra padding though");
            builder.append_value("Another string that's meant to be smaller than the previous value, though still need extra padding");
            let array = builder.finish();
            array.statistics().set_iter(
                array
                    .statistics()
                    .compute_all(&Stat::all().collect::<Vec<_>>())
                    .vortex_unwrap()
                    .into_iter(),
            );

            let layout = FlatLayoutStrategy::default()
                .write_stream(&ctx, sequence_writer, stream_only(array.to_array()))
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let result = layout
                .new_reader("".into(), segments)
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &root(),
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            assert_eq!(
                result.statistics().get_as::<String>(Stat::Min),
                // The typo is correct, we need this to be truncated.
                Some(Precision::Inexact(
                    "Another string that's meant to be smaller than the previous valu".to_string()
                ))
            );
            assert_eq!(
                result.statistics().get_as::<String>(Stat::Max),
                Some(Precision::Inexact(
                    "Long value to test that the statistics are actually truncated, j".to_string()
                ))
            );
        })
    }

    #[test]
    fn struct_array_round_trip() {
        block_on(async {
            let mut validity_builder = BooleanBufferBuilder::new(2);
            validity_builder.append(true);
            validity_builder.append(false);
            let validity_boolean_buffer = validity_builder.finish();
            let validity = Validity::Array(
                BoolArray::from_bool_buffer(validity_boolean_buffer.clone(), Validity::NonNullable)
                    .into_array(),
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
                let segments = TestSegments::default();
                let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
                let layout = FlatLayoutStrategy::default()
                    .write_stream(&ctx, sequence_writer, stream_only(array.to_array()))
                    .await
                    .unwrap();

                (layout, Arc::new(segments) as Arc<dyn SegmentSource>)
            };

            // We should be able to read the array we just wrote.
            let result: ArrayRef = layout
                .new_reader("".into(), segments)
                .unwrap()
                .projection_evaluation(
                    &(0..layout.row_count()),
                    &root(),
                    MaskFuture::new_true(layout.row_count().try_into().unwrap()),
                )
                .unwrap()
                .await
                .unwrap();

            assert_eq!(
                result.validity_mask().boolean_buffer(),
                AllOr::Some(&validity_boolean_buffer)
            );
            assert_eq!(
                result
                    .to_struct()
                    .field_by_name("a")
                    .unwrap()
                    .to_primitive()
                    .as_slice::<u64>(),
                &[1, 2]
            );
            assert_eq!(
                result
                    .to_struct()
                    .field_by_name("b")
                    .unwrap()
                    .to_primitive()
                    .as_slice::<u64>(),
                &[3, 4]
            );
        })
    }
}
