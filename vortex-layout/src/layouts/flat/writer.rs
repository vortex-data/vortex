use vortex_array::serde::SerializeOptions;
use vortex_array::stats::{Precision, Stat, StatsProvider};
use vortex_array::{Array, ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_scalar::{BinaryScalar, Utf8Scalar};

use crate::layouts::flat::FlatLayout;
use crate::layouts::zoned::{lower_bound, upper_bound};
use crate::segments::SegmentWriter;
use crate::writer::LayoutWriter;
use crate::{IntoLayout, LayoutRef, LayoutStrategy, LayoutWriterExt};

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
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(FlatLayoutWriter::new(ctx.clone(), dtype.clone(), self.clone()).boxed())
    }
}

/// Writer for a [`FlatLayout`].
pub struct FlatLayoutWriter {
    ctx: ArrayContext,
    dtype: DType,
    options: FlatLayoutStrategy,
    layout: Option<LayoutRef>,
}

impl FlatLayoutWriter {
    pub fn new(ctx: ArrayContext, dtype: DType, options: FlatLayoutStrategy) -> Self {
        Self {
            ctx,
            dtype,
            options,
            layout: None,
        }
    }
}

impl LayoutWriter for FlatLayoutWriter {
    fn push_chunk(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        assert_eq!(
            chunk.dtype(),
            &self.dtype,
            "Can't push chunks of the wrong dtype into a LayoutWriter. Pushed {} but expected {}.",
            chunk.dtype(),
            self.dtype
        );

        if self.layout.is_some() {
            vortex_bail!("FlatLayoutStrategy::push_batch called after finish");
        }
        let row_count = chunk.len() as u64;

        match chunk.dtype() {
            DType::Utf8(_) => {
                if let Some(sv) = chunk.statistics().get(Stat::Min) {
                    let (value, truncated) = lower_bound::<Utf8Scalar>(
                        chunk.dtype(),
                        sv.into_inner(),
                        self.options.max_variable_length_statistics_size,
                    )?;
                    if truncated {
                        chunk.statistics().set(Stat::Min, Precision::Inexact(value));
                    }
                }

                if let Some(sv) = chunk.statistics().get(Stat::Max) {
                    let (value, truncated) = upper_bound::<Utf8Scalar>(
                        chunk.dtype(),
                        sv.into_inner(),
                        self.options.max_variable_length_statistics_size,
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
                        self.options.max_variable_length_statistics_size,
                    )?;
                    if truncated {
                        chunk.statistics().set(Stat::Min, Precision::Inexact(value));
                    }
                }

                if let Some(sv) = chunk.statistics().get(Stat::Max) {
                    let (value, truncated) = upper_bound::<BinaryScalar>(
                        chunk.dtype(),
                        sv.into_inner(),
                        self.options.max_variable_length_statistics_size,
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

        let buffers = chunk.serialize(
            &self.ctx,
            &SerializeOptions {
                offset: 0,
                include_padding: self.options.include_padding,
            },
        )?;
        let segment_id = segment_writer.put(&buffers);

        self.layout =
            Some(FlatLayout::new(row_count, self.dtype.clone(), segment_id).into_layout());

        Ok(())
    }

    fn flush(&mut self, _segment_writer: &mut dyn SegmentWriter) -> VortexResult<()> {
        Ok(())
    }

    fn finish(&mut self, _segment_writer: &mut dyn SegmentWriter) -> VortexResult<LayoutRef> {
        self.layout
            .take()
            .ok_or_else(|| vortex_err!("FlatLayoutStrategy::finish called without push_batch"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::builders::{ArrayBuilder, VarBinViewBuilder};
    use vortex_array::stats::{Precision, Stat};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayContext};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability};
    use vortex_error::VortexUnwrap;
    use vortex_expr::ident;
    use vortex_mask::Mask;

    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::segments::{SegmentSource, TestSegments};
    use crate::writer::LayoutWriterExt;

    // Currently, flat layouts do not force compute stats during write, they only retain
    // pre-computed stats.
    #[should_panic]
    #[test]
    fn flat_stats() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let mut segments = TestSegments::default();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout =
                FlatLayoutWriter::new(ctx.clone(), array.dtype().clone(), Default::default())
                    .push_one(&mut segments, array.to_array())
                    .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

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
            let mut segments = TestSegments::default();
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

            let layout =
                FlatLayoutWriter::new(ctx.clone(), array.dtype().clone(), Default::default())
                    .push_one(&mut segments, array.to_array())
                    .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

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
