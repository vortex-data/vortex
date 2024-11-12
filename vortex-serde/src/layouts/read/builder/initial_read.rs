use core::ops::Range;

use bytes::{Bytes, BytesMut};
use flatbuffers::root;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::io::VortexReadAt;
use crate::layouts::{EOF_SIZE, INITIAL_READ_SIZE, MAGIC_BYTES, VERSION};
use vortex_flatbuffers::footer::Footer;

pub(crate) struct InitialRead {
    pub buf: Bytes,
    pub initial_read_offset: u64,
    pub fb_footer_byte_range: Range<usize>,
}

impl InitialRead {
    pub fn fb_footer(&self) -> VortexResult<Footer> {
        Ok(root::<Footer>(&self.buf[self.fb_footer_byte_range.clone()])?)
    }
}

pub(crate) async fn read_initial_bytes<R: VortexReadAt>(
    read: &R,
    file_size: u64,
) -> Result<InitialRead, VortexError> {
    if file_size < EOF_SIZE as u64 {
        vortex_bail!(
            "Malformed vortex file, size {} must be at least {}",
            file_size,
            EOF_SIZE,
        )
    }

    let read_size = INITIAL_READ_SIZE.min(file_size as usize);
    let mut buf = BytesMut::with_capacity(read_size);
    unsafe { buf.set_len(read_size) }

    let initial_read_offset = file_size - read_size as u64;
    buf = read.read_at_into(initial_read_offset, buf).await?;

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
    let footer_size = u16::from_le_bytes(
        buf[eof_loc + 2..eof_loc + 4]
            .try_into()
            .map_err(|e| vortex_err!("Footer size was not a u16 {e}"))?,
    ) as usize;
    if footer_size > eof_loc {
        vortex_bail!(
            "Malformed file, footer of size {} is too large to fit in initial read of size {} (file size {})",
            footer_size,
            read_size,
            file_size,
        )
    }

    let footer_loc = eof_loc - footer_size;
    let fb_footer_byte_range = footer_loc..eof_loc;
    let footer = root::<Footer>(&buf[fb_footer_byte_range.clone()])?;
    let schema_offset = footer.schema_offset();
    let layout_offset = footer.layout_offset();

    if layout_offset > initial_read_offset + footer_loc as u64 {
        vortex_bail!(
            "Layout must come before the Footer, got layout_offset {}, but footer starts at offset {}",
            layout_offset,
            initial_read_offset + footer_loc as u64,
        )
    }

    if layout_offset < schema_offset {
        vortex_bail!(
            "Schema must come before the Layout, got schema_offset {} and layout_offset {}",
            schema_offset,
            layout_offset,
        )
    }

    if initial_read_offset < schema_offset {
        // TODO: instead of bailing, we can just read more bytes.
        vortex_bail!(
            "Schema, layout, & footer must be in the initial read, got schema at {} and initial read from {}",
            schema_offset,
            initial_read_offset,
        )
    }

    Ok(InitialRead {
        buf: buf.freeze(),
        initial_read_offset,
        fb_footer_byte_range,
    })
}

#[cfg(test)]
mod tests {
    use crate::layouts::MAX_FOOTER_SIZE;

    use super::*;

    #[test]
    fn big_enough_initial_read() {
        assert!(INITIAL_READ_SIZE > EOF_SIZE + MAX_FOOTER_SIZE as usize);
    }
}
