use core::ops::Range;

use bytes::Bytes;
use flatbuffers::{root, root_unchecked};
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_flatbuffers::{footer, message};
use vortex_io::VortexReadAt;

use crate::{
    LayoutDeserializer, LayoutReader, LazyDType, RelativeLayoutCache, Scan, EOF_SIZE,
    INITIAL_READ_SIZE, MAGIC_BYTES, VERSION,
};

#[derive(Debug)]
pub struct InitialRead {
    /// The bytes from the initial read of the file, which is assumed (for now) to be sufficiently
    /// large to contain the schema and layout.
    pub buf: Bytes,
    /// The absolute byte offset representing the start of the initial read within the file.
    pub initial_read_offset: u64,
    /// The byte range within `buf` representing the Postscript flatbuffer.
    pub fb_postscript_byte_range: Range<usize>,
}

impl InitialRead {
    pub fn fb_postscript(&self) -> VortexResult<footer::Postscript> {
        Ok(unsafe {
            root_unchecked::<footer::Postscript>(&self.buf[self.fb_postscript_byte_range.clone()])
        })
    }

    /// The bytes of the `Layout` flatbuffer.
    pub fn fb_layout_byte_range(&self) -> VortexResult<Range<usize>> {
        let footer = self.fb_postscript()?;
        let layout_start = (footer.layout_offset() - self.initial_read_offset) as usize;
        let layout_end = self.fb_postscript_byte_range.start;
        Ok(layout_start..layout_end)
    }

    /// The `Layout` flatbuffer.
    pub fn fb_layout(&self) -> VortexResult<footer::Layout> {
        Ok(unsafe { root_unchecked::<footer::Layout>(&self.buf[self.fb_layout_byte_range()?]) })
    }

    /// The bytes of the `Schema` flatbuffer.
    pub fn fb_schema_byte_range(&self) -> VortexResult<Range<usize>> {
        let footer = self.fb_postscript()?;
        let schema_start = (footer.schema_offset() - self.initial_read_offset) as usize;
        let schema_end = (footer.layout_offset() - self.initial_read_offset) as usize;
        Ok(schema_start..schema_end)
    }

    pub fn lazy_dtype(&self) -> VortexResult<LazyDType> {
        // we validated the schema bytes at construction time
        unsafe {
            Ok(LazyDType::from_schema_bytes(
                self.buf.slice(self.fb_schema_byte_range()?),
            ))
        }
    }
}

pub fn read_layout_from_initial(
    initial_read: &InitialRead,
    layout_serde: &LayoutDeserializer,
    scan: Scan,
    message_cache: RelativeLayoutCache,
) -> VortexResult<Box<dyn LayoutReader>> {
    let layout_bytes = initial_read.buf.slice(initial_read.fb_layout_byte_range()?);
    let fb_loc = initial_read.fb_layout()?._tab.loc();
    layout_serde.read_layout(layout_bytes, fb_loc, scan, message_cache)
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

    let initial_read_offset = file_size - read_size as u64;
    let buf = read
        .read_byte_range(initial_read_offset, read_size as u64)
        .await?;

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

    // The footer MUST fit in the initial read.
    let ps_size = u16::from_le_bytes(
        buf[eof_loc + 2..eof_loc + 4]
            .try_into()
            .map_err(|e| vortex_err!("Footer size was not a u16 {e}"))?,
    ) as usize;
    if ps_size > eof_loc {
        vortex_bail!(
            "Malformed file, postscript of size {} is too large to fit in initial read of size {} (file size {})",
            ps_size,
            read_size,
            file_size,
        )
    }

    let ps_loc = eof_loc - ps_size;
    let fb_postscript_byte_range = ps_loc..eof_loc;

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

    if schema_offset < initial_read_offset {
        // TODO: instead of bailing, we can just read more bytes.
        vortex_bail!(
            "Schema, layout, & footer must be in the initial read, got schema at {} and initial read from {}",
            schema_offset,
            initial_read_offset,
        )
    }

    // validate the schema and layout
    let schema_loc = (schema_offset - initial_read_offset) as usize;
    let layout_loc = (layout_offset - initial_read_offset) as usize;
    root::<message::Schema>(&buf[schema_loc..layout_loc])?;
    root::<footer::Layout>(&buf[layout_loc..ps_loc])?;

    Ok(InitialRead {
        buf,
        initial_read_offset,
        fb_postscript_byte_range,
    })
}

#[cfg(test)]
mod tests {
    use crate::{EOF_SIZE, INITIAL_READ_SIZE, MAX_FOOTER_SIZE};

    #[test]
    fn big_enough_initial_read() {
        assert!(INITIAL_READ_SIZE > EOF_SIZE + MAX_FOOTER_SIZE as usize);
    }
}
