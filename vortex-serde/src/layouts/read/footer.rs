use bytes::{Bytes, BytesMut};
use flatbuffers::{root, root_unchecked};
use vortex_dtype::field::Field;
use vortex_dtype::flatbuffers::deserialize_and_project;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_flatbuffers::{footer, message as fb};

use crate::io::VortexReadAt;
use crate::layouts::read::cache::RelativeLayoutCache;
use crate::layouts::read::context::LayoutDeserializer;
use crate::layouts::read::{LayoutReader, Scan, INITIAL_READ_SIZE};
use crate::layouts::{EOF_SIZE, FOOTER_POSTSCRIPT_SIZE, MAGIC_BYTES, VERSION};
use crate::FLATBUFFER_SIZE_LENGTH;

/// A description of the contents of a Vortex file, including dtype and layouts.
///
/// In particular, this class is constructed from the last five sections of a Vortex file: the
/// schema (data type), footer, postscript, version, and magic bytes.
///
/// # File Format
/// ```text
/// ┌────────────────────────────┐
/// │                            │
/// │            Data            │
/// │    (Array IPC Messages)    │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │     Per-Column Metadata    │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │          Schema            │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │          Footer            │
/// │    (Layouts + Row Count)   │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │        Postscript          │
/// │  (Schema/Footer Offsets)   │
/// │        (32 bytes)          │
/// │                            │
/// ├────────────────────────────┤
/// │   Version Info (4 bytes)   │
/// ├────────────────────────────┤
/// │    Magic bytes (4 bytes)   │
/// └────────────────────────────┘
/// ```
#[derive(Debug)]
pub struct LayoutDescriptor {
    pub(crate) schema_offset: u64,
    pub(crate) footer_offset: u64,
    pub(crate) initial_read: Bytes,
    pub(crate) initial_read_offset: u64,
    pub(crate) layout_serde: LayoutDeserializer,
}

impl LayoutDescriptor {
    fn initial_read_layout_offset(&self) -> usize {
        (self.footer_offset - self.initial_read_offset) as usize
    }

    fn initial_read_schema_offset(&self) -> usize {
        (self.schema_offset - self.initial_read_offset) as usize
    }

    /// The total number of rows contained in this file.
    pub fn row_count(&self) -> VortexResult<u64> {
        Ok(self.fb_footer()?.row_count())
    }

    /// A [LayoutReader] which will read and produce one or more arrays from the data and metadata.
    pub fn layout(
        &self,
        scan: Scan,
        message_cache: RelativeLayoutCache,
    ) -> VortexResult<Box<dyn LayoutReader>> {
        let start_offset = self.initial_read_layout_offset();
        let end_offset = self.initial_read.len() - FOOTER_POSTSCRIPT_SIZE - EOF_SIZE;
        let footer_bytes = self
            .initial_read
            .slice(start_offset + FLATBUFFER_SIZE_LENGTH..end_offset);
        let fb_footer = unsafe { root_unchecked::<footer::Footer>(&footer_bytes) };

        let fb_layout = fb_footer
            .layout()
            .ok_or_else(|| vortex_err!("Footer must contain a layout"))?;
        let loc = fb_layout._tab.loc();
        self.layout_serde
            .read_layout(footer_bytes, loc, scan, message_cache)
    }

    pub fn dtype_bytes(&self) -> VortexResult<Bytes> {
        let start_offset = self.initial_read_schema_offset();
        let end_offset = self.initial_read_layout_offset();
        let bytes = self
            .initial_read
            .slice(start_offset + FLATBUFFER_SIZE_LENGTH..end_offset);
        // Run validation on dtype bytes
        self.fb_schema()?;
        Ok(bytes)
    }

    pub fn dtype(&self) -> VortexResult<DType> {
        DType::try_from(
            self.fb_schema()?
                .dtype()
                .ok_or_else(|| vortex_err!(InvalidSerde: "Schema missing DType"))?,
        )
    }

    pub fn projected_dtype(&self, projection: &[Field]) -> VortexResult<DType> {
        let fb_dtype = self
            .fb_schema()?
            .dtype()
            .ok_or_else(|| vortex_err!(InvalidSerde: "Schema missing DType"))?;
        deserialize_and_project(fb_dtype, projection)
    }

    fn fb_footer(&self) -> VortexResult<footer::Footer> {
        let start_offset = self.initial_read_layout_offset();
        let end_offset = self.initial_read.len() - FOOTER_POSTSCRIPT_SIZE - EOF_SIZE;
        let footer_bytes = &self.initial_read[start_offset + FLATBUFFER_SIZE_LENGTH..end_offset];
        Ok(root::<footer::Footer>(footer_bytes)?)
    }

    fn fb_schema(&self) -> VortexResult<fb::Schema> {
        let start_offset = self.initial_read_schema_offset();
        let end_offset = self.initial_read_layout_offset();
        let dtype_bytes = &self.initial_read[start_offset + FLATBUFFER_SIZE_LENGTH..end_offset];

        root::<fb::Message>(dtype_bytes)
            .map_err(|e| e.into())
            .and_then(|m| {
                m.header_as_schema()
                    .ok_or_else(|| vortex_err!("Message was not a schema"))
            })
    }
}

pub struct LayoutDescriptorReader {
    layout_serde: LayoutDeserializer,
}

impl LayoutDescriptorReader {
    pub fn new(layout_serde: LayoutDeserializer) -> Self {
        Self { layout_serde }
    }

    pub async fn read_footer<R: VortexReadAt>(
        &self,
        read: &R,
        file_size: u64,
    ) -> VortexResult<LayoutDescriptor> {
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

        let read_offset = file_size - read_size as u64;
        buf = read.read_at_into(read_offset, buf).await?;

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

        let ps = root::<footer::Postscript>(&buf[eof_loc - FOOTER_POSTSCRIPT_SIZE..eof_loc])?;

        Ok(LayoutDescriptor {
            schema_offset: ps.schema_offset(),
            footer_offset: ps.footer_offset(),
            initial_read: buf.freeze(),
            initial_read_offset: read_offset,
            layout_serde: self.layout_serde.clone(),
        })
    }
}
