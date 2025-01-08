use core::ops::Range;

use flatbuffers::{root, root_unchecked};
use vortex_buffer::{ByteBuffer, ByteBufferMut, ConstBuffer};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult, VortexUnwrap};
use vortex_flatbuffers::{dtype as fbd, footer};
use vortex_io::VortexReadAt;

use crate::{EOF_SIZE, INITIAL_READ_SIZE, MAGIC_BYTES, VERSION};

#[derive(Debug, Clone)]
pub struct InitialRead {
    /// The bytes from the initial read of the file, which is assumed (for now) to be sufficiently
    /// large to contain the schema and layout.
    // TODO(ngates): we should ensure the initial read, and therefore the flatbuffers, are
    //  8-byte aligned. But the writer doesn't guarantee this right now.
    pub buf: ConstBuffer<u8, 1>,
    /// The absolute byte offset representing the start of the initial read within the file.
    pub initial_read_offset: u64,
    /// The byte range within `buf` representing the Postscript flatbuffer.
    pub fb_postscript_byte_range: Range<usize>,
}

impl InitialRead {
    pub fn fb_postscript(&self) -> footer::Postscript {
        unsafe {
            root_unchecked::<footer::Postscript>(&self.buf[self.fb_postscript_byte_range.clone()])
        }
    }

    /// The bytes of the `Layout` flatbuffer.
    pub fn fb_layout_byte_range(&self) -> Range<usize> {
        let footer = self.fb_postscript();
        let layout_start = (footer.layout_offset() - self.initial_read_offset) as usize;
        let layout_end = self.fb_postscript_byte_range.start;
        layout_start..layout_end
    }

    /// The `Layout` flatbuffer.
    pub fn fb_layout(&self) -> footer::Layout {
        unsafe { root_unchecked::<footer::Layout>(&self.buf[self.fb_layout_byte_range()]) }
    }

    /// The bytes of the `Schema` flatbuffer.
    pub fn fb_schema_byte_range(&self) -> Range<usize> {
        let footer = self.fb_postscript();
        let schema_start = (footer.schema_offset() - self.initial_read_offset) as usize;
        let schema_end = (footer.layout_offset() - self.initial_read_offset) as usize;
        schema_start..schema_end
    }

    pub fn dtype(&self) -> DType {
        let dtype_buffer = self.buf.as_ref().slice(self.fb_schema_byte_range());
        let fb_dtype = unsafe { root_unchecked::<fbd::DType>(&dtype_buffer) };

        DType::try_from_view(fb_dtype, dtype_buffer.clone())
            .vortex_expect("Initial read must be able to provide valid flatbuffer DType")
    }
}

pub async fn read_initial_bytes<R: VortexReadAt>(
    read: &R,
    file_size: u64,
) -> VortexResult<InitialRead> {
    if file_size < EOF_SIZE as u64 {
        vortex_bail!(
            "Malformed vortex file, size {} must be at least {}",
            file_size,
            EOF_SIZE,
        )
    }

    let read_size = INITIAL_READ_SIZE.min(file_size as usize);

    let mut initial_read_offset = file_size - read_size as u64;

    let mut buf = ByteBuffer::from(
        read.read_byte_range(initial_read_offset, read_size as u64)
            .await?,
    );

    let eof_loc = read_size - EOF_SIZE;
    let magic_bytes_loc = eof_loc + (EOF_SIZE - MAGIC_BYTES.len());

    let magic_number = &buf[magic_bytes_loc..];
    if magic_number != MAGIC_BYTES {
        vortex_bail!("Malformed file, invalid magic bytes, got {magic_number:?}")
    }

    let version = u16::from_le_bytes(
        buf[eof_loc..eof_loc + 2]
            .try_into()
            .map_err(|e| vortex_err!("Version was not a u16 {e}"))?,
    );
    if version != VERSION {
        vortex_bail!("Malformed file, unsupported version {version}")
    }

    let ps_size =
        u16::from_le_bytes(buf[eof_loc + 2..eof_loc + 4].try_into().vortex_unwrap()) as usize;

    if ps_size > eof_loc {
        vortex_bail!(
            "Malformed file, postscript of size {} is too large to fit in initial read of size {} (file size {})",
            ps_size,
            read_size,
            file_size,
        )
    }

    let mut ps_loc = eof_loc - ps_size;
    let mut fb_postscript_byte_range = ps_loc..eof_loc;

    // we validate the footer here
    let postscript = root::<footer::Postscript>(&buf[fb_postscript_byte_range.clone()])?;
    let schema_offset = postscript.schema_offset();
    let layout_offset = postscript.layout_offset();

    if layout_offset > initial_read_offset + ps_loc as u64 {
        vortex_bail!(
            "Layout must come before the Footer, got layout_offset {}, but footer starts at offset {}",
            layout_offset,
            initial_read_offset + ps_loc as u64,
        )
    }

    if layout_offset < schema_offset {
        vortex_bail!(
            "Schema must come before the Layout, got schema_offset {} and layout_offset {}",
            schema_offset,
            layout_offset,
        )
    }

    // If the schema is not in the initial read, we need to read more bytes.
    // Note that this will perform a copy of the initial read, since we are prepending the schema.
    if schema_offset < initial_read_offset {
        let mut buf_builder = ByteBufferMut::with_capacity((file_size - schema_offset).try_into()?);
        let prefix_bytes = initial_read_offset - schema_offset;
        let leftover = read.read_byte_range(schema_offset, prefix_bytes).await?;

        let prefix_bytes: usize = prefix_bytes.try_into()?;

        buf_builder.extend_from_slice(&leftover);
        buf_builder.extend_from_slice(&buf);

        buf = buf_builder.freeze();
        // Reset the absolute offsets to account for the new data that was fetched and
        // prepended to the buffer.
        initial_read_offset = schema_offset;
        fb_postscript_byte_range.start += prefix_bytes;
        fb_postscript_byte_range.end += prefix_bytes;
        ps_loc += prefix_bytes;
    }

    // validate the schema and layout
    let schema_loc = (schema_offset - initial_read_offset) as usize;
    let layout_loc = (layout_offset - initial_read_offset) as usize;
    root::<fbd::DType>(&buf[schema_loc..layout_loc])?;
    root::<footer::Layout>(&buf[layout_loc..ps_loc])?;

    Ok(InitialRead {
        buf: buf.try_into()?,
        initial_read_offset,
        fb_postscript_byte_range,
    })
}

#[cfg(test)]
mod tests {
    use flatbuffers::FlatBufferBuilder;
    use vortex_buffer::ByteBufferMut;
    use vortex_flatbuffers::footer::PostscriptBuilder;

    use crate::{
        read_initial_bytes, EOF_SIZE, INITIAL_READ_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION,
    };

    #[test]
    fn big_enough_initial_read() {
        assert!(INITIAL_READ_SIZE > EOF_SIZE + MAX_FOOTER_SIZE as usize);
    }

    #[tokio::test]
    async fn test_read_initial_bytes() {
        let postscript = make_postscript(0, 1024);
        let mut buf = ByteBufferMut::with_capacity(INITIAL_READ_SIZE);
        // write the "schema"
        buf.extend_from_slice(&[0; 1024]);
        // write the "layout"
        buf.extend_from_slice(&[0; 1024]);

        let postscript_start = buf.len();
        // Write the postscript flatbuffer
        buf.extend_from_slice(&postscript);

        // Write the EOF.
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(postscript.len() as u16).to_le_bytes());
        buf.extend_from_slice(&MAGIC_BYTES);

        let buf = buf.freeze().into_inner();

        assert!(buf.len() <= INITIAL_READ_SIZE);
        let initial_read = read_initial_bytes(&buf, buf.len() as u64).await.unwrap();

        assert_eq!(initial_read.initial_read_offset, 0);
        assert_eq!(
            initial_read.fb_postscript_byte_range,
            postscript_start..(postscript_start + postscript.len())
        );
        assert_eq!(initial_read.fb_schema_byte_range(), 0..1024,);
        assert_eq!(initial_read.fb_layout_byte_range(), 1024..2048);
    }

    #[tokio::test]
    async fn test_read_initial_bytes_large() {
        // We create a Postscript flatbuffer message, with a schema offset that
        // is before the INITIAL_READ_SIZE to test that the `read_initial_bytes` will
        // refetch as necessary.

        // Write the Postscript at the very end of the buffer.
        let postscript = make_postscript(0, 1024);

        // We create a virtual footer that includes a schema offset, followed by a layout offset,
        // both before INITIAL_READ_SIZE.
        let mut buf = ByteBufferMut::with_capacity(2 * INITIAL_READ_SIZE);

        // Write a bunch of zeros to pad.
        let postscript_start = 2 * INITIAL_READ_SIZE - EOF_SIZE - postscript.len();
        buf.extend_from_slice(&vec![0; postscript_start]);
        assert_eq!(buf.len(), postscript_start);

        // Write the footer flatbuffer message.
        buf.extend_from_slice(&postscript);

        // Write the 8-byte EOF structure which contains the version,
        // the postscript, and the magic bytes.
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(postscript.len() as u16).to_le_bytes());
        buf.extend_from_slice(&MAGIC_BYTES);

        let buf = buf.freeze().into_inner();

        assert_eq!(buf.len(), 2 * INITIAL_READ_SIZE);

        let initial_read = read_initial_bytes(&buf, buf.len() as u64).await.unwrap();

        assert_eq!(initial_read.initial_read_offset, 0);
        assert_eq!(
            initial_read.fb_postscript_byte_range,
            postscript_start..(postscript_start + postscript.len())
        );
        assert_eq!(initial_read.fb_schema_byte_range(), 0..1024);
        assert_eq!(initial_read.fb_layout_byte_range(), 1024..postscript_start);
    }

    fn make_postscript(schema_offset: u64, layout_offset: u64) -> Vec<u8> {
        let mut fbb = FlatBufferBuilder::new();
        let mut postscript_builder = PostscriptBuilder::new(&mut fbb);
        postscript_builder.add_schema_offset(schema_offset);
        postscript_builder.add_layout_offset(layout_offset);
        let root_offset = postscript_builder.finish();
        fbb.finish(root_offset, None);
        fbb.finished_data().to_vec()
    }
}
