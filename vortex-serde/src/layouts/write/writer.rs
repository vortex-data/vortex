use std::{io, mem};

use flatbuffers::FlatBufferBuilder;
use futures::{Stream, TryStreamExt};
use vortex::array::{ChunkedArray, StructArray};
use vortex::stream::ArrayStream;
use vortex::{Array, ArrayDType as _, IntoArray};
use vortex_buffer::io_buf::IoBuf;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect as _, VortexResult};
use vortex_flatbuffers::WriteFlatBuffer;

use crate::io::VortexWrite;
use crate::layouts::write::footer::{Footer, Postscript};
use crate::layouts::write::layouts::Layout;
use crate::layouts::write::metadata_accumulators::{new_metadata_accumulator, MetadataAccumulator};
use crate::layouts::{EOF_SIZE, MAGIC_BYTES, VERSION};
use crate::stream_writer::ByteRange;
use crate::MessageWriter;

pub struct LayoutWriter<W> {
    msgs: MessageWriter<W>,

    row_count: u64,
    dtype: Option<DType>,
    column_metadata: Vec<Box<dyn MetadataAccumulator>>,
}

impl<W: VortexWrite> LayoutWriter<W> {
    pub fn new(write: W) -> Self {
        LayoutWriter {
            msgs: MessageWriter::new(write),
            dtype: None,
            column_metadata: Vec::new(),
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
        S: Stream<Item = VortexResult<Array>> + Unpin + ArrayStream,
    {
        let size_hint = stream.size_hint().0;
        let metadata = match self.column_metadata.get_mut(column_idx) {
            None => {
                self.column_metadata
                    .push(new_metadata_accumulator(size_hint, stream.dtype()));

                assert_eq!(
                    self.column_metadata.len(),
                    column_idx + 1,
                    "write_column_metadata must be called in order by column index! got column index {} but column chunks has {} columns",
                    column_idx,
                    self.column_metadata.len()
                );

                self.column_metadata
                    .last_mut()
                    .vortex_expect("column chunks cannot be empty, just pushed")
            }
            Some(x) => x,
        };

        let mut byte_offsets = Vec::with_capacity(size_hint + 1);
        byte_offsets.push(self.msgs.tell());
        while let Some(chunk) = stream.try_next().await? {
            metadata.push_chunk(&chunk)?;
            self.msgs.write_batch(chunk).await?;
            byte_offsets.push(self.msgs.tell());
        }

        metadata.push_batch_byte_offsets(byte_offsets);

        Ok(())
    }

    async fn write_metadata_arrays(&mut self) -> VortexResult<Layout> {
        let mut column_layouts = Vec::with_capacity(self.column_metadata.len());
        for column_metadata in mem::take(&mut self.column_metadata) {
            let (mut chunk_layouts, metadata_array) =
                column_metadata.into_layouts_and_metadata()?;

            let dtype_begin = self.msgs.tell();
            self.msgs.write_dtype(metadata_array.dtype()).await?;
            let dtype_end = self.msgs.tell();
            self.msgs.write_batch(metadata_array.into_array()).await?;

            chunk_layouts.push_front(Layout::inlined_schema(
                vec![Layout::flat(ByteRange::new(dtype_end, self.msgs.tell()))],
                ByteRange::new(dtype_begin, dtype_end),
            ));
            column_layouts.push(Layout::chunked(chunk_layouts.into(), true));
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
    }
}
