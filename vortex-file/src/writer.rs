// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;

use futures::TryStreamExt;
use futures::executor::block_on;
use futures::future::try_join;
use vortex_array::ArrayContext;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter, ArrayStreamExt};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt};
use vortex_io::VortexWrite;
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::segments::SequenceWriter;
use vortex_layout::{LayoutContext, LayoutStrategy, LocalExecutor};

use crate::footer::{FileStatistics, FooterFlatBufferWriter, Postscript, PostscriptSegment};
use crate::segments::writer::SerialSegmentWriter;
use crate::{EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION, WriteStrategyBuilder};

/// Configure a new writer, which can eventually be used to write an [`ArrayStream`] into a sink that implements [`VortexWrite`].
///
/// Unless overridden, the default [write strategy][crate::WriteStrategyBuilder] will be used with no
/// additional configuration.
pub struct VortexWriteOptions {
    strategy: Arc<dyn LayoutStrategy>,
    exclude_dtype: bool,
    max_variable_length_statistics_size: usize,
    file_statistics: Vec<Stat>,
}

impl Default for VortexWriteOptions {
    fn default() -> Self {
        Self {
            strategy: WriteStrategyBuilder::new()
                .with_executor(Arc::new(LocalExecutor))
                .build(),
            exclude_dtype: false,
            file_statistics: PRUNING_STATS.to_vec(),
            max_variable_length_statistics_size: 64,
        }
    }
}

impl VortexWriteOptions {
    /// Replace the default layout strategy with the provided one.
    pub fn with_strategy(mut self, strategy: Arc<dyn LayoutStrategy>) -> Self {
        self.strategy = strategy;
        self
    }

    /// Exclude the DType from the Vortex file. You must provide the DType to the reader.
    // TODO(ngates): Should we store some sort of DType checksum to make sure the one passed at
    //  read-time is sane? I guess most layouts will have some reasonable validation.
    pub fn exclude_dtype(mut self) -> Self {
        self.exclude_dtype = true;
        self
    }

    /// Configure which statistics to compute at the file-level.
    pub fn with_file_statistics(mut self, file_statistics: Vec<Stat>) -> Self {
        self.file_statistics = file_statistics;
        self
    }
}

impl VortexWriteOptions {
    /// Write to an `ObjectStore` using the provided `VortexWrite` implementation.
    #[cfg(feature = "object_store")]
    pub async fn write_object_store<S: ArrayStream + Unpin + Send + 'static>(
        self,
        object_store: &Arc<dyn object_store::ObjectStore>,
        path: &object_store::path::Path,
        stream: S,
    ) -> VortexResult<()> {
        use futures::future::FutureExt;
        use vortex_io::ObjectStoreWriter;

        self.write(
            ObjectStoreWriter::new(object_store.clone(), path).await?,
            stream,
        )
        .boxed()
        .await?
        .shutdown()
        .await?;
        Ok(())
    }

    /// Perform a blocking single-threaded write of the provided stream of `Array`.
    pub fn write_blocking<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        write: W,
        stream: S,
    ) -> VortexResult<W> {
        block_on(self.write(write, stream))
    }

    /// Perform an async write of the provided stream of `Array`.
    pub async fn write<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        write: W,
        stream: S,
    ) -> VortexResult<W> {
        // Set up a Context to capture the encodings used in the file.
        let ctx = ArrayContext::empty();

        let dtype = stream.dtype().clone();
        let (segment_writer, flusher) = SerialSegmentWriter::create();
        let sequence_writer = SequenceWriter::new(Box::new(segment_writer));

        let stream = stream.try_filter(|chunk| future::ready(!chunk.is_empty()));

        let stream = sequence_writer.new_sequential(ArrayStreamExt::boxed(
            ArrayStreamAdapter::new(dtype.clone(), stream),
        ));

        let (file_stats, stream) = accumulate_stats(
            stream,
            self.file_statistics.clone().into(),
            self.max_variable_length_statistics_size,
        );

        // First we write the magic number
        let mut write = futures::io::Cursor::new(write);
        write.write_all(MAGIC_BYTES).await?;

        let io_fut = flusher.flush(write);
        let compute_fut = self.strategy.write_stream(&ctx, sequence_writer, stream);
        let (layout, (mut write, segment_specs)) = try_join(compute_fut, io_fut).await?;

        // We write our footer components in order of least likely to be needed to most likely.
        // DType is the least likely to be needed, as many readers may provide this from an
        // external source.
        let dtype_segment = if self.exclude_dtype {
            None
        } else {
            Some(self.write_flatbuffer(&mut write, &dtype).await?)
        };

        let layout_ctx = LayoutContext::empty();
        let layout_segment = self
            .write_flatbuffer(&mut write, &layout.flatbuffer_writer(&layout_ctx))
            .await?;

        let statistics_segment = if self.file_statistics.is_empty() {
            None
        } else {
            let file_statistics = FileStatistics(file_stats.stats_sets().into());
            Some(self.write_flatbuffer(&mut write, &file_statistics).await?)
        };

        let footer_segment = self
            .write_flatbuffer(
                &mut write,
                &FooterFlatBufferWriter {
                    ctx,
                    layout_ctx,
                    segment_specs: segment_specs.into(),
                },
            )
            .await?;

        // Assemble the postscript, and write it manually to avoid any framing.
        let postscript = Postscript {
            dtype: dtype_segment,
            layout: layout_segment,
            statistics: statistics_segment,
            footer: footer_segment,
        };
        let postscript_buffer = postscript.write_flatbuffer_bytes();
        if postscript_buffer.len() > MAX_FOOTER_SIZE as usize {
            vortex_bail!(
                "Postscript is too large ({} bytes); max postscript size is {}",
                postscript_buffer.len(),
                MAX_FOOTER_SIZE
            );
        }
        let postscript_len = u16::try_from(postscript_buffer.len())
            .vortex_expect("Postscript already verified to fit into u16");
        write.write_all(postscript_buffer).await?;

        // And finally, the EOF 8-byte footer.
        let mut eof = [0u8; EOF_SIZE];
        eof[0..2].copy_from_slice(&VERSION.to_le_bytes());
        eof[2..4].copy_from_slice(&postscript_len.to_le_bytes());
        eof[4..8].copy_from_slice(&MAGIC_BYTES);
        write.write_all(eof).await?;

        write.flush().await?;

        Ok(write.into_inner())
    }

    async fn write_flatbuffer<W: VortexWrite, F: FlatBufferRoot + WriteFlatBuffer>(
        &self,
        write: &mut futures::io::Cursor<W>,
        flatbuffer: &F,
    ) -> VortexResult<PostscriptSegment> {
        let layout_offset = write.position();
        write.write_all(flatbuffer.write_flatbuffer_bytes()).await?;
        Ok(PostscriptSegment {
            offset: layout_offset,
            length: u32::try_from(write.position() - layout_offset)
                .map_err(|_| vortex_err!("segment length exceeds maximum u32"))?,
            alignment: FlatBuffer::alignment(),
        })
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[cfg(feature = "object_store")]
    fn create_test_array(length: usize) -> vortex_array::ArrayRef {
        use vortex_array::IntoArray;
        use vortex_array::arrays::{BoolArray, PrimitiveArray, StructArray, VarBinArray};
        use vortex_array::validity::Validity;
        use vortex_buffer::Buffer;
        use vortex_dtype::{DType, Nullability};

        let id_values: Vec<i64> = (1..=length).map(|i| i as i64).collect();
        let x_values: Vec<i32> = (1..=length).map(|i| i as i32).collect();
        let y_values: Vec<i32> = (1..=length).map(|i| -(i as i32)).collect();
        let score_values: Vec<f32> = (0..length).map(|i| 0.5 + i as f32).collect();
        let temperature_values: Vec<f64> = (0..length).map(|i| 98.6 + (i as f64) * 0.5).collect();
        let counter_a_values: Vec<u32> = (1..=length).map(|i| (i * 10) as u32).collect();
        let counter_b_values: Vec<u64> = (1..=length).map(|i| (i * 100) as u64).collect();
        let small_val_values: Vec<u8> = (1..=length).map(|i| i as u8).collect();
        let medium_val_values: Vec<u16> = (1..=length).map(|i| (i * 1000) as u16).collect();
        let negative_small_values: Vec<i8> = (1..=length)
            .map(|i| -((i % i8::MAX as usize) as i8))
            .collect();
        let negative_medium_values: Vec<i32> = (1..=length).map(|i| -(i as i32)).collect();
        let metadata_1_values: Vec<i32> = (0..length).map(|i| (111 * (i + 1)) as i32).collect();
        let metadata_2_values: Vec<i32> = (0..length).map(|i| (555 + 111 * i) as i32).collect();
        let metadata_3_values: Vec<i32> = (0..length).map(|i| (999 + 370 * i) as i32).collect();
        let timestamp_values: Vec<i64> = (1..=length).map(|i| (i * 10) as i64).collect();
        let duration_ms_values: Vec<u32> = (0..length).map(|i| (100 + i * 150) as u32).collect();
        let percentage_values: Vec<f32> = (0..length).map(|i| 0.25 * (i + 1) as f32).collect();
        let ratio_values: Vec<f64> = (0..length).map(|i| 1.5 + i as f64).collect();
        let field_26_values: Vec<i32> = (0..length).map(|i| (26 + i) as i32).collect();
        let field_27_values: Vec<i64> = (0..length).map(|i| (270 + i) as i64).collect();
        let field_28_values: Vec<f32> = (0..length).map(|i| 2.8 + 0.01 * i as f32).collect();
        let field_29_values: Vec<u32> = (0..length).map(|i| (290 + i) as u32).collect();
        let field_31_values: Vec<i16> = (0..length).map(|i| (31 + i) as i16).collect();
        let field_32_values: Vec<f64> = (0..length).map(|i| 3.2 + 0.01 * i as f64).collect();
        let field_33_values: Vec<u64> = (0..length).map(|i| (330 + i) as u64).collect();
        let field_35_values: Vec<u8> = (0..length).map(|i| (35 + i) as u8).collect();
        let field_37_values: Vec<i32> = (0..length).map(|i| (370 + i) as i32).collect();
        let field_38_values: Vec<f32> = (0..length).map(|i| 38.0 + 0.1 * i as f32).collect();
        let field_39_values: Vec<u16> = (0..length).map(|i| (3900 + i) as u16).collect();
        let field_41_values: Vec<i8> = (0..length)
            .map(|i| -((i % i8::MAX as usize) as i8))
            .collect();
        let field_42_values: Vec<f64> = (0..length).map(|i| 42.0 + 0.1 * i as f64).collect();
        let field_43_values: Vec<i64> = (0..length).map(|i| (4300 + i) as i64).collect();
        let field_45_values: Vec<u32> = (0..length).map(|i| (450 + i) as u32).collect();
        let field_47_values: Vec<i32> = (0..length).map(|i| (47 + i) as i32).collect();
        let field_48_values: Vec<f32> = (0..length).map(|i| 4.8 + 0.01 * i as f32).collect();
        let field_49_values: Vec<u64> = (0..length).map(|i| (490 + i) as u64).collect();

        let names = ["Alice", "Bob", "Charlie", "David"];
        let descriptions = ["First user", "Second user", "Third user", "Fourth user"];
        let tags = ["admin,user", "user", "moderator,user", "guest"];
        let field_34_vals = ["value34a", "value34b", "value34c", "value34d"];
        let field_40_vals = ["item40-1", "item40-2", "item40-3", "item40-4"];
        let field_46_vals = ["text46a", "text46b", "text46c", "text46d"];

        let name_values: Vec<&str> = (0..length).map(|i| names[i % 4]).collect();
        let description_values: Vec<&str> = (0..length).map(|i| descriptions[i % 4]).collect();
        let tags_values: Vec<&str> = (0..length).map(|i| tags[i % 4]).collect();
        let field_34_values: Vec<&str> = (0..length).map(|i| field_34_vals[i % 4]).collect();
        let field_40_values: Vec<&str> = (0..length).map(|i| field_40_vals[i % 4]).collect();
        let field_46_values: Vec<&str> = (0..length).map(|i| field_46_vals[i % 4]).collect();

        let is_active_values: Vec<bool> = (0..length).map(|i| i % 2 == 0).collect();
        let is_verified_values: Vec<bool> = (0..length).map(|i| (i + 1) % 2 == 0).collect();
        let flag_a_values: Vec<bool> = (0..length).map(|i| i % 4 != 2).collect();
        let flag_b_values: Vec<bool> = (0..length).map(|i| i % 4 == 2).collect();
        let field_30_values: Vec<bool> = (0..length).map(|i| i % 2 == 0 || i % 4 == 3).collect();
        let field_36_values: Vec<bool> = (0..length).map(|i| i % 2 == 1).collect();
        let field_44_values: Vec<bool> = (0..length).map(|i| i % 4 != 3).collect();
        let field_50_values: Vec<bool> = (0..length).map(|i| i % 4 == 3).collect();

        StructArray::from_fields(&[
            (
                "id",
                PrimitiveArray::new(Buffer::from_iter(id_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "x",
                PrimitiveArray::new(Buffer::from_iter(x_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "y",
                PrimitiveArray::new(Buffer::from_iter(y_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "score",
                PrimitiveArray::new(Buffer::from_iter(score_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "temperature",
                PrimitiveArray::new(Buffer::from_iter(temperature_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "is_active",
                BoolArray::from_iter(is_active_values).into_array(),
            ),
            (
                "is_verified",
                BoolArray::from_iter(is_verified_values).into_array(),
            ),
            (
                "counter_a",
                PrimitiveArray::new(Buffer::from_iter(counter_a_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "counter_b",
                PrimitiveArray::new(Buffer::from_iter(counter_b_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "small_val",
                PrimitiveArray::new(Buffer::from_iter(small_val_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "medium_val",
                PrimitiveArray::new(Buffer::from_iter(medium_val_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "negative_small",
                PrimitiveArray::new(
                    Buffer::from_iter(negative_small_values),
                    Validity::NonNullable,
                )
                .into_array(),
            ),
            (
                "negative_medium",
                PrimitiveArray::new(
                    Buffer::from_iter(negative_medium_values),
                    Validity::NonNullable,
                )
                .into_array(),
            ),
            (
                "name",
                VarBinArray::from_vec(name_values, DType::Utf8(Nullability::NonNullable))
                    .into_array(),
            ),
            (
                "description",
                VarBinArray::from_vec(description_values, DType::Utf8(Nullability::NonNullable))
                    .into_array(),
            ),
            (
                "tags",
                VarBinArray::from_vec(tags_values, DType::Utf8(Nullability::NonNullable))
                    .into_array(),
            ),
            (
                "metadata_1",
                PrimitiveArray::new(Buffer::from_iter(metadata_1_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "metadata_2",
                PrimitiveArray::new(Buffer::from_iter(metadata_2_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "metadata_3",
                PrimitiveArray::new(Buffer::from_iter(metadata_3_values), Validity::NonNullable)
                    .into_array(),
            ),
            ("flag_a", BoolArray::from_iter(flag_a_values).into_array()),
            ("flag_b", BoolArray::from_iter(flag_b_values).into_array()),
            (
                "timestamp",
                PrimitiveArray::new(Buffer::from_iter(timestamp_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "duration_ms",
                PrimitiveArray::new(Buffer::from_iter(duration_ms_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "percentage",
                PrimitiveArray::new(Buffer::from_iter(percentage_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "ratio",
                PrimitiveArray::new(Buffer::from_iter(ratio_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_26",
                PrimitiveArray::new(Buffer::from_iter(field_26_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_27",
                PrimitiveArray::new(Buffer::from_iter(field_27_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_28",
                PrimitiveArray::new(Buffer::from_iter(field_28_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_29",
                PrimitiveArray::new(Buffer::from_iter(field_29_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_30",
                BoolArray::from_iter(field_30_values).into_array(),
            ),
            (
                "field_31",
                PrimitiveArray::new(Buffer::from_iter(field_31_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_32",
                PrimitiveArray::new(Buffer::from_iter(field_32_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_33",
                PrimitiveArray::new(Buffer::from_iter(field_33_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_34",
                VarBinArray::from_vec(field_34_values, DType::Utf8(Nullability::NonNullable))
                    .into_array(),
            ),
            (
                "field_35",
                PrimitiveArray::new(Buffer::from_iter(field_35_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_36",
                BoolArray::from_iter(field_36_values).into_array(),
            ),
            (
                "field_37",
                PrimitiveArray::new(Buffer::from_iter(field_37_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_38",
                PrimitiveArray::new(Buffer::from_iter(field_38_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_39",
                PrimitiveArray::new(Buffer::from_iter(field_39_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_40",
                VarBinArray::from_vec(field_40_values, DType::Utf8(Nullability::NonNullable))
                    .into_array(),
            ),
            (
                "field_41",
                PrimitiveArray::new(Buffer::from_iter(field_41_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_42",
                PrimitiveArray::new(Buffer::from_iter(field_42_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_43",
                PrimitiveArray::new(Buffer::from_iter(field_43_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_44",
                BoolArray::from_iter(field_44_values).into_array(),
            ),
            (
                "field_45",
                PrimitiveArray::new(Buffer::from_iter(field_45_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_46",
                VarBinArray::from_vec(field_46_values, DType::Utf8(Nullability::NonNullable))
                    .into_array(),
            ),
            (
                "field_47",
                PrimitiveArray::new(Buffer::from_iter(field_47_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_48",
                PrimitiveArray::new(Buffer::from_iter(field_48_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_49",
                PrimitiveArray::new(Buffer::from_iter(field_49_values), Validity::NonNullable)
                    .into_array(),
            ),
            (
                "field_50",
                BoolArray::from_iter(field_50_values).into_array(),
            ),
        ])
        .unwrap()
        .into_array()
    }

    #[cfg(feature = "object_store")]
    #[tokio::test]
    #[rstest::rstest]
    #[case(10)]
    #[case(100_000)]
    #[case(1_000_000)]
    async fn test_write_object_store(#[case] array_length: usize) -> anyhow::Result<()> {
        use object_store::local::LocalFileSystem;
        use object_store::path::Path;
        use vortex_scan::ScanBuilder;

        use crate::VortexOpenOptions;

        let tempdir = tempfile::tempdir()?;
        let object_store = Arc::new(LocalFileSystem::new_with_prefix(tempdir.path())?) as _;
        let location = &Path::from("file.vx");

        let vortex_array = create_test_array(array_length);

        let dtype = vortex_array.dtype().clone();

        let stream = ArrayStreamAdapter::new(
            dtype.clone(),
            futures::stream::iter(std::iter::once(Ok(vortex_array.clone()))),
        );

        VortexWriteOptions::default()
            .write_object_store(&object_store, location, stream)
            .await?;

        dbg!(object_store.head(location).await?.size);

        let vx_file = VortexOpenOptions::file()
            .open_object_store(&object_store, location.as_ref())
            .await?;
        assert_eq!(&dtype, vx_file.dtype());

        let output = ScanBuilder::new(vx_file.layout_reader()?)
            .into_tokio_array_stream()?
            .read_all()
            .await?;

        assert_eq!(vortex_array.len(), output.len());
        assert_eq!(
            vortex_array.display_values().to_string(),
            output.display_values().to_string()
        );
        Ok(())
    }
}
