use std::{io, mem};

use flatbuffers::FlatBufferBuilder;
use futures::TryStreamExt;
use itertools::Itertools;
use vortex_array::array::{ChunkedArray, StructArray};
use vortex_array::stats::{ArrayStatistics, Stat};
use vortex_array::stream::ArrayStream;
use vortex_array::{ArrayDType as _, ArrayData, ArrayLen};
use vortex_buffer::io_buf::IoBuf;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect as _, VortexResult};
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_io::VortexWrite;
use vortex_ipc::messages::writer::MessageWriter;
use vortex_ipc::messages::IPCSchema;
use vortex_ipc::stream_writer::ByteRange;

use crate::write::metadata_accumulators::{new_metadata_accumulator, MetadataAccumulator};
use crate::write::postscript::Postscript;
use crate::{LayoutSpec, EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION};

const STATS_TO_WRITE: &[Stat] = &[
    Stat::Min,
    Stat::Max,
    Stat::TrueCount,
    Stat::NullCount,
    Stat::RunCount,
    Stat::IsConstant,
    Stat::IsSorted,
    Stat::IsStrictSorted,
    Stat::UncompressedSizeInBytes,
];

pub struct VortexFileWriter<W> {
    msgs: MessageWriter<W>,

    row_count: u64,
    dtype: Option<DType>,
    column_writers: Vec<ColumnWriter>,
}

impl<W: VortexWrite> VortexFileWriter<W> {
    pub fn new(write: W) -> Self {
        VortexFileWriter {
            msgs: MessageWriter::new(write),
            dtype: None,
            column_writers: Vec::new(),
            row_count: 0,
        }
    }

    pub async fn write_array_columns(self, array: ArrayData) -> VortexResult<Self> {
        if let Ok(chunked) = ChunkedArray::try_from(array.clone()) {
            self.write_array_columns_stream(chunked.array_stream())
                .await
        } else {
            self.write_array_columns_stream(array.into_array_stream())
                .await
        }
    }

    pub async fn write_array_columns_stream<S: ArrayStream + Unpin>(
        mut self,
        mut array_stream: S,
    ) -> VortexResult<Self> {
        match self.dtype {
            None => self.dtype = Some(array_stream.dtype().clone()),
            Some(ref sd) => {
                if sd != array_stream.dtype() {
                    vortex_bail!(
                        "Expected all arrays in the stream to have the same dtype {}, found {}",
                        sd,
                        array_stream.dtype()
                    )
                }
            }
        }

        while let Some(columns) = array_stream.try_next().await? {
            let st = StructArray::try_from(columns)?;
            self.row_count += st.len() as u64;
            for (i, field) in st.children().enumerate() {
                if let Ok(chunked_array) = ChunkedArray::try_from(field.clone()) {
                    self.write_column_chunks(chunked_array.array_stream(), i)
                        .await?
                } else {
                    self.write_column_chunks(field.into_array_stream(), i)
                        .await?
                }
            }
        }

        Ok(self)
    }

    async fn write_column_chunks<S>(&mut self, stream: S, column_idx: usize) -> VortexResult<()>
    where
        S: ArrayStream + Unpin,
    {
        let column_writer = match self.column_writers.get_mut(column_idx) {
            None => {
                self.column_writers.push(ColumnWriter::new(stream.dtype()));

                assert_eq!(
                    self.column_writers.len(),
                    column_idx + 1,
                    "write_column_chunks must be called in order by column index! got column index {} but column chunks has {} columns",
                    column_idx,
                    self.column_writers.len()
                );

                self.column_writers
                    .last_mut()
                    .vortex_expect("column chunks cannot be empty, just pushed")
            }
            Some(x) => x,
        };

        column_writer.write_chunks(stream, &mut self.msgs).await
    }

    async fn write_metadata_arrays(&mut self) -> VortexResult<LayoutSpec> {
        let mut column_layouts = Vec::with_capacity(self.column_writers.len());
        for column_writer in mem::take(&mut self.column_writers) {
            column_layouts.push(
                column_writer
                    .write_metadata(self.row_count, &mut self.msgs)
                    .await?,
            );
        }

        Ok(LayoutSpec::column(column_layouts, self.row_count))
    }

    pub async fn finalize(mut self) -> VortexResult<W> {
        let top_level_layout = self.write_metadata_arrays().await?;
        let schema_offset = self.msgs.tell();

        // we want to write raw flatbuffers from here on out, not messages
        let mut writer = self.msgs.into_inner();

        // write the schema, and get the start offset of the next section (layout)
        let layout_offset = {
            let dtype = self
                .dtype
                .take()
                .ok_or_else(|| vortex_err!("Schema should be written by now"))?;
            // we write an IPCSchema instead of a DType, which allows us to evolve / add to the schema later
            // these bytes get deserialized as message::Schema
            // NB: we don't wrap the IPCSchema in an IPCMessage, because we record the lengths/offsets in the footer
            let schema = IPCSchema(&dtype);
            let schema_len = write_fb_raw(&mut writer, schema).await?;
            schema_offset + schema_len
        };

        // write the layout
        write_fb_raw(&mut writer, top_level_layout).await?;

        let footer = Postscript::try_new(schema_offset, layout_offset)?;
        let footer_len = write_fb_raw(&mut writer, footer).await?;
        if footer_len > MAX_FOOTER_SIZE as u64 {
            vortex_bail!(
                "Footer is too large ({} bytes); max footer size is {}",
                footer_len,
                MAX_FOOTER_SIZE
            );
        }
        let footer_len = footer_len as u16;

        let mut eof = [0u8; EOF_SIZE];
        eof[0..2].copy_from_slice(&VERSION.to_le_bytes());
        eof[2..4].copy_from_slice(&footer_len.to_le_bytes());
        eof[4..8].copy_from_slice(&MAGIC_BYTES);

        writer.write_all(eof).await?;
        Ok(writer)
    }
}

/// Write a flatbuffer to a writer and return the number of bytes written.
async fn write_fb_raw<W: VortexWrite, F: WriteFlatBuffer>(
    writer: &mut W,
    fb: F,
) -> io::Result<u64> {
    let mut fbb = FlatBufferBuilder::new();
    let ps_fb = fb.write_flatbuffer(&mut fbb);
    fbb.finish_minimal(ps_fb);

    let (buffer, buffer_begin) = fbb.collapse();
    let buffer_end = buffer.len();

    let bytes = buffer.slice_owned(buffer_begin..buffer_end);
    writer.write_all(bytes).await?;
    Ok((buffer_end - buffer_begin) as u64)
}

struct ColumnWriter {
    metadata: Box<dyn MetadataAccumulator>,
    batch_byte_offsets: Vec<Vec<u64>>,
    batch_row_offsets: Vec<Vec<u64>>,
}

impl ColumnWriter {
    fn new(dtype: &DType) -> Self {
        Self {
            metadata: new_metadata_accumulator(dtype),
            batch_byte_offsets: Vec::new(),
            batch_row_offsets: Vec::new(),
        }
    }

    async fn write_chunks<W: VortexWrite, S: ArrayStream + Unpin>(
        &mut self,
        mut stream: S,
        msgs: &mut MessageWriter<W>,
    ) -> VortexResult<()> {
        let mut offsets = Vec::with_capacity(stream.size_hint().0 + 1);
        offsets.push(msgs.tell());
        let mut row_offsets = Vec::with_capacity(stream.size_hint().0 + 1);
        row_offsets.push(
            self.batch_row_offsets
                .last()
                .and_then(|bro| bro.last())
                .copied()
                .unwrap_or(0),
        );

        let mut rows_written = row_offsets[0];

        while let Some(chunk) = stream.try_next().await? {
            rows_written += chunk.len() as u64;

            // accumulate the stats for the stats table
            self.metadata.push_chunk(&chunk);

            // clear the stats that we don't want to serialize into the file
            chunk.statistics().retain_only(STATS_TO_WRITE);

            msgs.write_batch(chunk).await?;
            offsets.push(msgs.tell());
            row_offsets.push(rows_written);
        }

        self.batch_byte_offsets.push(offsets);
        self.batch_row_offsets.push(row_offsets);

        Ok(())
    }

    async fn write_metadata<W: VortexWrite>(
        self,
        row_count: u64,
        msgs: &mut MessageWriter<W>,
    ) -> VortexResult<LayoutSpec> {
        let data_chunks = self
            .batch_byte_offsets
            .into_iter()
            .zip(self.batch_row_offsets.into_iter())
            .flat_map(|(byte_offsets, row_offsets)| {
                byte_offsets
                    .into_iter()
                    .tuple_windows::<(_, _)>()
                    .map(|(begin, end)| ByteRange::new(begin, end))
                    .zip(
                        row_offsets
                            .into_iter()
                            .tuple_windows::<(_, _)>()
                            .map(|(begin, end)| end - begin),
                    )
                    .map(|(range, len)| LayoutSpec::flat(range, len))
            });

        if let Some(metadata_array) = self.metadata.into_array()? {
            let expected_n_data_chunks = metadata_array.len();

            let dtype_begin = msgs.tell();
            msgs.write_dtype_raw(metadata_array.dtype()).await?;
            let dtype_end = msgs.tell();
            msgs.write_batch(metadata_array).await?;
            let metadata_array_end = msgs.tell();

            let layouts = [LayoutSpec::inlined_schema(
                vec![LayoutSpec::flat(
                    ByteRange::new(dtype_end, metadata_array_end),
                    expected_n_data_chunks as u64,
                )],
                expected_n_data_chunks as u64,
                ByteRange::new(dtype_begin, dtype_end),
            )]
            .into_iter()
            .chain(data_chunks)
            .collect::<Vec<_>>();

            if layouts.len() != expected_n_data_chunks + 1 {
                vortex_bail!(
                    "Expected {} layouts based on row offsets, found {} based on byte offsets",
                    expected_n_data_chunks + 1,
                    layouts.len()
                );
            }
            Ok(LayoutSpec::chunked(layouts, row_count, true))
        } else {
            Ok(LayoutSpec::chunked(data_chunks.collect(), row_count, false))
        }
    }
}

#[cfg(test)]
mod tests {
    use flatbuffers::FlatBufferBuilder;
    use futures_executor::block_on;
    use vortex_array::array::{PrimitiveArray, StructArray, VarBinArray};
    use vortex_array::stats::PRUNING_STATS;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayData;
    use vortex_flatbuffers::WriteFlatBuffer;

    use crate::write::postscript::Postscript;
    use crate::write::writer::STATS_TO_WRITE;
    use crate::{VortexFileWriter, V1_FOOTER_FBS_SIZE};

    #[test]
    fn write_columns() {
        let strings = VarBinArray::from(vec!["ab", "foo", "bar", "baz"]);
        let numbers = PrimitiveArray::from(vec![1u32, 2, 3, 4]);
        let st = StructArray::try_new(
            ["strings".into(), "numbers".into()].into(),
            vec![strings.into_array(), numbers.into_array()],
            4,
            Validity::NonNullable,
        )
        .unwrap();
        let buf = Vec::new();
        let mut writer = VortexFileWriter::new(buf);
        writer = block_on(async { writer.write_array_columns(st.into_array()).await }).unwrap();
        let written = block_on(async { writer.finalize().await }).unwrap();
        assert!(!written.is_empty());
    }

    #[test]
    fn footer_size() {
        let footer = Postscript::try_new(1000000u64, 1100000u64).unwrap();
        let mut fbb = FlatBufferBuilder::new();
        let footer_fb = footer.write_flatbuffer(&mut fbb);
        fbb.finish_minimal(footer_fb);
        let (buffer, buffer_begin) = fbb.collapse();
        let buffer_end = buffer.len();

        assert_eq!(buffer[buffer_begin..buffer_end].len(), V1_FOOTER_FBS_SIZE);
    }

    #[test]
    fn stats_to_write() {
        for stat in PRUNING_STATS {
            assert!(STATS_TO_WRITE.contains(stat));
        }
    }
}
