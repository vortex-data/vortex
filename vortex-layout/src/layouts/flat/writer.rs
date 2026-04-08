// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::ArrayContext;
use vortex_array::ArrayId;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::normalize::NormalizeOptions;
use vortex_array::normalize::Operation;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarTruncation;
use vortex_array::scalar::lower_bound;
use vortex_array::scalar::upper_bound;
use vortex_array::serde::SerializeOptions;
use vortex_array::stats::StatsSetRef;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::runtime::Handle;
use vortex_session::registry::ReadContext;
use vortex_utils::aliases::hash_set::HashSet;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::flat::FlatLayout;
use crate::layouts::flat::flat_layout_inline_array_node;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;

#[derive(Clone)]
pub struct FlatLayoutStrategy {
    /// Whether to include padding for memory-mapped reads.
    pub include_padding: bool,
    /// Maximum length of variable length statistics
    pub max_variable_length_statistics_size: usize,
    /// Optional set of allowed array encodings for normalization.
    /// If None, then all are allowed.
    pub allowed_encodings: Option<HashSet<ArrayId>>,
}

impl Default for FlatLayoutStrategy {
    fn default() -> Self {
        Self {
            include_padding: true,
            max_variable_length_statistics_size: 64,
            allowed_encodings: None,
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

    /// Set the allowed array encodings for normalization.
    pub fn with_allow_encodings(mut self, allow_encodings: HashSet<ArrayId>) -> Self {
        self.allowed_encodings = Some(allow_encodings);
        self
    }
}

fn truncate_scalar_stat<F: Fn(Scalar) -> Option<(Scalar, bool)>>(
    statistics: StatsSetRef<'_>,
    stat: Stat,
    truncation: F,
) {
    if let Some(sv) = statistics.get(stat) {
        if let Some((truncated_value, truncated)) = truncation(sv.into_inner()) {
            if truncated && let Some(v) = truncated_value.into_value() {
                statistics.set(stat, Precision::Inexact(v));
            }
        } else {
            statistics.clear(stat)
        }
    }
}

#[async_trait]
impl LayoutStrategy for FlatLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        _eof: SequencePointer,
        _handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let ctx = ctx.clone();
        let Some(chunk) = stream.next().await else {
            vortex_bail!("flat layout needs a single chunk");
        };
        let (sequence_id, chunk) = chunk?;

        let row_count = chunk.len() as u64;

        match chunk.dtype() {
            DType::Utf8(n) => {
                truncate_scalar_stat(chunk.statistics(), Stat::Min, |v| {
                    lower_bound(
                        BufferString::from_scalar(v)
                            .vortex_expect("utf8 scalar must be a BufferString"),
                        self.max_variable_length_statistics_size,
                        *n,
                    )
                });
                truncate_scalar_stat(chunk.statistics(), Stat::Max, |v| {
                    upper_bound(
                        BufferString::from_scalar(v)
                            .vortex_expect("utf8 scalar must be a BufferString"),
                        self.max_variable_length_statistics_size,
                        *n,
                    )
                });
            }
            DType::Binary(n) => {
                truncate_scalar_stat(chunk.statistics(), Stat::Min, |v| {
                    lower_bound(
                        ByteBuffer::from_scalar(v)
                            .vortex_expect("binary scalar must be a ByteBuffer"),
                        self.max_variable_length_statistics_size,
                        *n,
                    )
                });
                truncate_scalar_stat(chunk.statistics(), Stat::Max, |v| {
                    upper_bound(
                        ByteBuffer::from_scalar(v)
                            .vortex_expect("binary scalar must be a ByteBuffer"),
                        self.max_variable_length_statistics_size,
                        *n,
                    )
                });
            }
            _ => {}
        }

        let chunk = if let Some(allowed) = &self.allowed_encodings {
            chunk.normalize(&mut NormalizeOptions {
                allowed,
                operation: Operation::Error,
            })?
        } else {
            chunk
        };

        let buffers = chunk.serialize(
            &ctx,
            &SerializeOptions {
                offset: 0,
                include_padding: self.include_padding,
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
            ReadContext::new(ctx.to_ids()),
            array_node,
        )
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        // FlatLayoutStrategy is a leaf strategy with no child strategies and no buffering
        0
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::DictArray;
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
    use vortex_array::expr::stats::Stat;
    use vortex_array::expr::stats::StatsProviderExt;
    use vortex_array::validity::Validity;
    use vortex_array::vtable::VTable;
    use vortex_buffer::BitBufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_mask::AllOr;
    use vortex_mask::Mask;
    use vortex_utils::aliases::hash_set::HashSet;

    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    // Currently, flat layouts do not force compute stats during write, they only retain
    // pre-computed stats.
    #[should_panic]
    #[test]
    fn flat_stats() {
        block_on(|handle| async {
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let array = PrimitiveArray::new(buffer![1, 2, 3, 4, 5], Validity::AllValid);
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    handle,
                )
                .await
                .unwrap();

            let result = layout
                .new_reader("".into(), segments, &SESSION)
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
        block_on(|handle| async {
            let ctx = ArrayContext::empty();
            let segments = Arc::new(TestSegments::default());
            let (ptr, eof) = SequenceId::root().split();
            let mut builder =
                VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::NonNullable), 2);
            builder.append_value("Long value to test that the statistics are actually truncated, it needs a bit of extra padding though");
            builder.append_value("Another string that's meant to be smaller than the previous value, though still need extra padding");
            let array = builder.finish();
            array.statistics().set_iter(
                array
                    .statistics()
                    .compute_all(&Stat::all().collect::<Vec<_>>())
                    .vortex_expect("stats computation should succeed for test array")
                    .into_iter(),
            );

            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    ctx,
                    Arc::<TestSegments>::clone(&segments),
                    array.into_array().to_array_stream().sequenced(ptr),
                    eof,
                    handle,
                )
                .await
                .unwrap();

            let result = layout
                .new_reader("".into(), segments, &SESSION)
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
                    // spellchecker:ignore-next-line
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
        block_on(|handle| async {
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
                        ctx,
                        Arc::<TestSegments>::clone(&segments),
                        array.into_array().to_array_stream().sequenced(ptr),
                        eof,
                        handle,
                    )
                    .await
                    .unwrap();

                (layout, segments)
            };

            // We should be able to read the array we just wrote.
            let result: ArrayRef = layout
                .new_reader("".into(), segments, &SESSION)
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
                result.validity_mask().unwrap().bit_buffer(),
                AllOr::Some(&validity_boolean_buffer)
            );
            assert_eq!(
                result
                    .to_struct()
                    .unmasked_field_by_name("a")
                    .unwrap()
                    .to_primitive()
                    .as_slice::<u64>(),
                &[1, 2]
            );
            assert_eq!(
                result
                    .to_struct()
                    .unmasked_field_by_name("b")
                    .unwrap()
                    .to_primitive()
                    .as_slice::<u64>(),
                &[3, 4]
            );
        })
    }

    #[test]
    fn flat_invalid_array_fails() -> VortexResult<()> {
        block_on(|handle| async {
            let prim: PrimitiveArray = (0..10).collect();
            let filter = prim.filter(Mask::from_indices(10, vec![2, 3]))?;

            let ctx = ArrayContext::empty();

            // Write the array into a byte buffer.
            let (layout, _segments) = {
                let segments = Arc::new(TestSegments::default());
                let (ptr, eof) = SequenceId::root().split();
                // Disallow all encodings so filter arrays fail normalization immediately.
                let allowed = HashSet::default();
                let layout = FlatLayoutStrategy::default()
                    .with_allow_encodings(allowed)
                    .write_stream(
                        ctx,
                        Arc::<TestSegments>::clone(&segments),
                        filter.into_array().to_array_stream().sequenced(ptr),
                        eof,
                        handle,
                    )
                    .await;

                (layout, segments)
            };

            let err = layout.expect_err("expected error");
            assert!(
                err.to_string()
                    .contains("normalize forbids encoding (vortex.filter)"),
                "unexpected error: {err}"
            );

            Ok(())
        })
    }

    #[test]
    fn flat_valid_array_writes() -> VortexResult<()> {
        block_on(|handle| async {
            let codes: PrimitiveArray = (0u32..10).collect();
            let values: PrimitiveArray = (0..10).collect();
            let dict = DictArray::new(codes.into_array(), values.into_array());

            let ctx = ArrayContext::empty();

            // Write the array into a byte buffer.
            let (layout, _segments) = {
                let segments = Arc::new(TestSegments::default());
                let (ptr, eof) = SequenceId::root().split();
                // Only allow the dict encoding; canonical primitive children remain permitted.
                let mut allowed = HashSet::default();
                allowed.insert(Dict.id());
                let layout = FlatLayoutStrategy::default()
                    .with_allow_encodings(allowed)
                    .write_stream(
                        ctx,
                        Arc::<TestSegments>::clone(&segments),
                        dict.into_array().to_array_stream().sequenced(ptr),
                        eof,
                        handle,
                    )
                    .await;

                (layout, segments)
            };

            assert!(layout.is_ok());

            Ok(())
        })
    }
}
