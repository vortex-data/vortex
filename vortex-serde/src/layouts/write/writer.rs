use std::collections::VecDeque;
use std::{io, mem};

use ahash::HashMapExt;
use flatbuffers::FlatBufferBuilder;
use futures::{Stream, TryStreamExt};
use vortex::array::{ChunkedArray, StructArray};
use vortex::stats::{ArrayStatistics, Stat};
use vortex::stream::ArrayStream;
use vortex::validity::Validity;
use vortex::{Array, ArrayDType, IntoArray};
use vortex_buffer::io_buf::IoBuf;
use vortex_buffer::{Buffer, BufferString};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_scalar::{Scalar, ScalarValue};

use super::footer::{Footer, Postscript};
use crate::io::VortexWrite;
use crate::layouts::write::layouts::Layout;
use crate::layouts::{EOF_SIZE, MAGIC_BYTES, METADATA_FIELD_NAMES, VERSION};
use crate::stream_writer::ByteRange;
use crate::MessageWriter;

pub struct LayoutWriter<W> {
    msgs: MessageWriter<W>,

    row_count: u64,
    dtype: Option<DType>,
    column_chunks: Vec<ColumnChunkAccumulator>,
}

impl<W: VortexWrite> LayoutWriter<W> {
    pub fn new(write: W) -> Self {
        LayoutWriter {
            msgs: MessageWriter::new(write),
            dtype: None,
            column_chunks: Vec::new(),
            row_count: 0,
        }
    }

    pub async fn write_array_columns(self, array: Array) -> VortexResult<Self> {
        if let Ok(chunked) = ChunkedArray::try_from(&array) {
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
            let st = StructArray::try_from(&columns)?;
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

    async fn write_column_chunks<S>(&mut self, mut stream: S, column_idx: usize) -> VortexResult<()>
    where
        S: Stream<Item = VortexResult<Array>> + Unpin,
    {
        let size_hint = stream.size_hint().0;
        let accumulator = match self.column_chunks.get_mut(column_idx) {
            None => {
                self.column_chunks
                    .push(ColumnChunkAccumulator::new(size_hint));

                assert_eq!(
                    self.column_chunks.len(),
                    column_idx + 1,
                    "write_column_chunks must be called in order by column index! got column index {} but column chunks has {} columns",
                    column_idx,
                    self.column_chunks.len()
                );

                self.column_chunks
                    .last_mut()
                    .vortex_expect("column chunks cannot be empty, just pushed")
            }
            Some(x) => x,
        };
        let mut n_rows_written = *accumulator
            .row_offsets
            .last()
            .vortex_expect("row offsets cannot be empty by construction");

        let mut byte_offsets = Vec::with_capacity(size_hint);
        byte_offsets.push(self.msgs.tell());

        while let Some(chunk) = stream.try_next().await? {
            for stat in PRUNING_STATS {
                accumulator.push_stat(stat, chunk.statistics().compute(stat));
            }

            n_rows_written += chunk.len() as u64;
            accumulator.push_row_offset(n_rows_written);

            self.msgs.write_batch(chunk).await?;
            byte_offsets.push(self.msgs.tell());
        }
        accumulator.push_batch_byte_offsets(byte_offsets);

        Ok(())
    }

    async fn write_metadata_arrays(&mut self) -> VortexResult<Layout> {
        let mut column_layouts = Vec::with_capacity(self.column_chunks.len());
        for column_accumulator in mem::take(&mut self.column_chunks) {
            let (mut chunks, metadata_array) = column_accumulator.into_chunks_and_metadata()?;

            let dtype_begin = self.msgs.tell();
            self.msgs.write_dtype(metadata_array.dtype()).await?;
            let dtype_end = self.msgs.tell();
            self.msgs.write_batch(metadata_array.into_array()).await?;
            chunks.push_front(Layout::inlined_schema(
                vec![Layout::flat(ByteRange::new(dtype_end, self.msgs.tell()))],
                ByteRange::new(dtype_begin, dtype_end),
            ));
            column_layouts.push(Layout::chunked(chunks.into(), true));
        }

        Ok(Layout::column(column_layouts))
    }

    async fn write_footer(&mut self, footer: Footer) -> VortexResult<Postscript> {
        let schema_offset = self.msgs.tell();
        self.msgs
            .write_dtype(
                &self
                    .dtype
                    .take()
                    .ok_or_else(|| vortex_err!("Schema should be written by now"))?,
            )
            .await?;
        let footer_offset = self.msgs.tell();
        self.msgs.write_message(footer).await?;
        Ok(Postscript::new(schema_offset, footer_offset))
    }

    pub async fn finalize(mut self) -> VortexResult<W> {
        let top_level_layout = self.write_metadata_arrays().await?;
        let ps = self
            .write_footer(Footer::new(top_level_layout, self.row_count))
            .await?;

        let mut w = self.msgs.into_inner();
        w = write_fb_raw(w, ps).await?;

        let mut eof = [0u8; EOF_SIZE];
        eof[0..2].copy_from_slice(&VERSION.to_le_bytes());
        eof[4..8].copy_from_slice(&MAGIC_BYTES);
        w.write_all(eof).await?;
        Ok(w)
    }
}

async fn write_fb_raw<W: VortexWrite, F: WriteFlatBuffer>(mut writer: W, fb: F) -> io::Result<W> {
    let mut fbb = FlatBufferBuilder::new();
    let ps_fb = fb.write_flatbuffer(&mut fbb);
    fbb.finish_minimal(ps_fb);
    let (buffer, buffer_begin) = fbb.collapse();
    let buffer_end = buffer.len();
    writer
        .write_all(buffer.slice_owned(buffer_begin..buffer_end))
        .await?;
    Ok(writer)
}

struct ColumnChunkAccumulator<> {
    pub dtype: DType,
    pub row_offsets: Vec<u64>,
    pub batch_byte_offsets: Vec<Vec<u64>>,
    pub minima: Vec<Option<ScalarValue>>,
    pub maxima: Vec<Option<ScalarValue>>,
    pub null_counts: Vec<Option<u64>>,
    pub true_counts: Vec<Option<u64>>,
}

impl ColumnChunkAccumulator {
    pub fn new(size_hint: usize, dtype: DType) -> Self {
        let mut row_offsets = Vec::with_capacity(size_hint + 1);
        row_offsets.push(0);
        Self {
            dtype,
            row_offsets,
            batch_byte_offsets: Vec::new(),
            minima: Vec::with_capacity(size_hint),
            maxima: Vec::with_capacity(size_hint),
            null_counts: Vec::with_capacity(size_hint),
            true_counts: Vec::with_capacity(size_hint),
        }
    }

    fn push_row_offset(&mut self, row_offset: u64) {
        self.row_offsets.push(row_offset);
    }

    fn push_batch_byte_offsets(&mut self, batch_byte_offsets: Vec<u64>) {
        self.batch_byte_offsets.push(batch_byte_offsets);
    }

    fn push_stat(&mut self, stat: Stat, value: Option<Scalar>) -> VortexResult<()> {
        if matches!(stat, Stat::Min | Stat::Max) {
            if let Some(value) = value {
                if !value.value().is_instance_of(&self.dtype) {
                    vortex_bail!("Expected all min/max values to have dtype {}, got {}", self.dtype, value.dtype());
                }
            }
        }

        Ok(match stat {
            Stat::Min => self.minima.push(value.map(|v| v.into_value())),
            Stat::Max => self.maxima.push(value.map(|v| v.into_value())),
            Stat::NullCount => self.null_counts.push(value.map_or(0, |v| {
                v.into_value()
                    .as_pvalue()
                    .vortex_expect("null count is a primitive value")
                    .and_then(|v| v.as_u64())
            })),
            Stat::TrueCount => self.true_counts.push(value.map_or(0, |v| {
                v.into_value()
                    .as_pvalue()
                    .vortex_expect("true count is a primitive value")
                    .and_then(|v| v.as_u64())
            })),
            _ => vortex_bail!("Unsupported pruning stat: {stat}"),
        })
    }

    fn into_chunks_and_metadata(mut self) -> VortexResult<(VecDeque<Layout>, Array)> {
        // we don't need the last row offset; that's just the total number of rows
        let length = self.row_offsets.len() - 1;
        self.row_offsets.truncate(length);

        let chunks: VecDeque<Layout> = self
            .batch_byte_offsets
            .iter()
            .flat_map(|byte_offsets| {
                byte_offsets
                    .iter()
                    .zip(byte_offsets.iter().skip(1))
                    .map(|(begin, end)| Layout::flat(ByteRange::new(*begin, *end)))
            })
            .collect();

        if chunks.len() != self.row_offsets.len() {
            vortex_bail!(
                "Expected {} chunks based on row offsets, found {} based on byte offsets",
                self.row_offsets.len(),
                chunks.len()
            );
        }

        let mut names = METADATA_FIELD_NAMES.clone();
        let mut fields = vec![self.row_offsets.into_array()];

        for stat in PRUNING_STATS {
            let values = self.pruning_stats.entry(stat).or_default();
            if values.len() != length {
                vortex_bail!(
                    "Expected {} values for stat {}, found {}",
                    length,
                    stat,
                    values.len()
                );
            }

            let Some(dtype) = values
                .iter()
                .filter(|v| v.is_some())
                .flatten()
                .map(|v| v.dtype())
                .next()
            else {
                // no point in writing all nulls
                continue;
            };
            let dtype = dtype.as_nullable();
            let values = values
                .iter()
                .map(|v| {
                    v.as_ref()
                        .map(|s| s.cast(&dtype).map(|s| s.into_value()))
                        .unwrap_or_else(|| Ok(ScalarValue::Null))
                })
                .collect::<VortexResult<Vec<_>>>()?;

            names.push(format!("{stat}").to_lowercase().into());
            fields.push(Array::from_scalar_values(dtype, values)?);
        }

        Ok((
            chunks,
            StructArray::try_new(names.into(), fields, length, Validity::NonNullable)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use flatbuffers::FlatBufferBuilder;
    use futures_executor::block_on;
    use vortex::array::{PrimitiveArray, StructArray, VarBinArray};
    use vortex::validity::Validity;
    use vortex::IntoArray;
    use vortex_flatbuffers::WriteFlatBuffer;

    use crate::layouts::write::footer::Postscript;
    use crate::layouts::{LayoutWriter, FOOTER_POSTSCRIPT_SIZE};

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
        let mut writer = LayoutWriter::new(buf);
        writer = block_on(async { writer.write_array_columns(st.into_array()).await }).unwrap();
        let written = block_on(async { writer.finalize().await }).unwrap();
        assert!(!written.is_empty());
    }

    #[test]
    fn postscript_size() {
        let ps = Postscript::new(1000000u64, 1100000u64);
        let mut fbb = FlatBufferBuilder::new();
        let ps_fb = ps.write_flatbuffer(&mut fbb);
        fbb.finish_minimal(ps_fb);
        let (buffer, buffer_begin) = fbb.collapse();
        let buffer_end = buffer.len();

        assert_eq!(
            buffer[buffer_begin..buffer_end].len(),
            FOOTER_POSTSCRIPT_SIZE
        );
        assert_eq!(buffer[buffer_begin..buffer_end].len(), 32);
    }
}
