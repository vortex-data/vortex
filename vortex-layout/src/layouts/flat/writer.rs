use futures::StreamExt;
use vortex_array::serde::SerializeOptions;
use vortex_array::stats::{Precision, Stat, StatsProvider};
use vortex_array::{Array, ArrayContext};
use vortex_dtype::DType;
use vortex_error::vortex_bail;
use vortex_scalar::{BinaryScalar, Utf8Scalar};

use crate::layouts::flat::FlatLayout;
use crate::layouts::zoned::{lower_bound, upper_bound};
use crate::segments::SequenceWriter;
use crate::{IntoLayout, LayoutStrategy, SendableLayoutWriter, SendableSequentialStream};

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

impl LayoutStrategy for FlatLayoutStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        mut stream: SendableSequentialStream,
    ) -> SendableLayoutWriter {
        let ctx = ctx.clone();
        let options = self.clone();
        let sequence_writer = sequence_writer.clone();
        Box::pin(async move {
            let Some(chunk) = stream.next().await else {
                vortex_bail!("flat layout needs a single chunk");
            };
            let (sequence_id, chunk) = chunk?;

            let row_count = chunk.len() as u64;

            match chunk.dtype() {
                DType::Utf8(_) => {
                    if let Some(sv) = chunk.statistics().get(Stat::Min) {
                        let (value, truncated) = lower_bound::<Utf8Scalar>(
                            chunk.dtype(),
                            sv.into_inner(),
                            options.max_variable_length_statistics_size,
                        )?;
                        if truncated {
                            chunk.statistics().set(Stat::Min, Precision::Inexact(value));
                        }
                    }

                    if let Some(sv) = chunk.statistics().get(Stat::Max) {
                        let (value, truncated) = upper_bound::<Utf8Scalar>(
                            chunk.dtype(),
                            sv.into_inner(),
                            options.max_variable_length_statistics_size,
                        )?;
                        if let Some(upper_bound) = value {
                            if truncated {
                                chunk
                                    .statistics()
                                    .set(Stat::Max, Precision::Inexact(upper_bound));
                            }
                        } else {
                            chunk.statistics().clear(Stat::Max)
                        }
                    }
                }
                DType::Binary(_) => {
                    if let Some(sv) = chunk.statistics().get(Stat::Min) {
                        let (value, truncated) = lower_bound::<BinaryScalar>(
                            chunk.dtype(),
                            sv.into_inner(),
                            options.max_variable_length_statistics_size,
                        )?;
                        if truncated {
                            chunk.statistics().set(Stat::Min, Precision::Inexact(value));
                        }
                    }

                    if let Some(sv) = chunk.statistics().get(Stat::Max) {
                        let (value, truncated) = upper_bound::<BinaryScalar>(
                            chunk.dtype(),
                            sv.into_inner(),
                            options.max_variable_length_statistics_size,
                        )?;
                        if let Some(upper_bound) = value {
                            if truncated {
                                chunk
                                    .statistics()
                                    .set(Stat::Max, Precision::Inexact(upper_bound));
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
            Ok(FlatLayout::new(row_count, stream.dtype().clone(), segment_id).into_layout())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use futures::{StreamExt, stream};
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::builders::{ArrayBuilder, VarBinViewBuilder};
    use vortex_array::stats::{Precision, Stat};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayContext, ArrayRef};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability};
    use vortex_error::VortexUnwrap;
    use vortex_expr::ident;
    use vortex_mask::Mask;

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::{SegmentSource, TestSegments};
    use crate::sequence::SequenceId;
    use crate::{LayoutStrategy, SendableSequentialStream, SequentialStreamAdapter};

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
            let segments = Arc::new(TestSegments::default());
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutStrategy::default()
                .write_stream(&ctx, segments.clone(), stream_only(array.to_array()))
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = segments;

            let result = layout
                .new_reader(&"".into(), &segments, &ctx)
                .unwrap()
                .projection_evaluation(&(0..layout.row_count()), &ident())
                .unwrap()
                .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
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
            let segments = Arc::new(TestSegments::default());
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
                .write_stream(
                    &ctx,
                    &array.dtype().clone(),
                    segments.clone(),
                    stream::once(
                        async move { Ok((SequenceId::root().downgrade(), array.to_array())) },
                    )
                    .boxed(),
                )
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = segments;

            let result = layout
                .new_reader(&"".into(), &segments, &ctx)
                .unwrap()
                .projection_evaluation(&(0..layout.row_count()), &ident())
                .unwrap()
                .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
                .await
                .unwrap();

            assert_eq!(
                result.statistics().get_as::<String>(Stat::Min),
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
}
