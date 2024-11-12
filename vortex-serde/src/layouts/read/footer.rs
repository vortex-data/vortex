use bytes::{Bytes, BytesMut};
use flatbuffers::{root, root_unchecked};
use vortex_dtype::field::Field;
use vortex_dtype::flatbuffers::deserialize_and_project;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_flatbuffers::{footer, message as fb};
use vortex_schema::projection::Projection;

use crate::io::VortexReadAt;
use crate::layouts::read::cache::RelativeLayoutCache;
use crate::layouts::read::context::LayoutDeserializer;
use crate::layouts::read::{LayoutReader, Scan, INITIAL_READ_SIZE};
use crate::layouts::{EOF_SIZE, FOOTER_FBS_SIZE, MAGIC_BYTES, VERSION};
use crate::FBS_TABLE_PREFIX_LENGTH;

use super::LazilyDeserializedDType;

/// A description of the reified contents of a Vortex file, including dtype and layouts.
/// 
/// Note that the per-column statistics coming after the data is a writer implementation detail,
/// rather than part of the spec. Additionally, because Layouts make the file essentially self-describing,
/// the statistics need not even be Array IPC messages (though they are currently).
///
/// The file format specification requires only that:
/// 
/// 1. Data is written first, followed by...
/// 2. An optional Schema, which if present is a valid DType flatbuffer, and is followed by...
/// 3. The Layout, which is a valid Layout flatbuffer, and is followed by...
/// 4. The Footer, which is a valid Footer flatbuffer, and is followed by...
/// 5. The End-of-File marker, which is 8 bytes, and contains the u16 version, u16 footer length, and 4 magic bytes.
///
/// In particular, this class is constructed from the last four boxes from the diagram below: the
/// schema, layout, footer, and the EOF.
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
/// │   Per-Column Statistics    │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │          Schema            │
/// |     (DType Flatbuffer)     │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │     Layout Flatbuffer      │
/// │                            │
/// ├────────────────────────────┤
/// │                            │
/// │     Footer Flatbuffer      │
/// │  (Schema & Layout Offsets) │
/// │                            │
/// ├────────────────────────────┤
/// │     8-byte End of File     │
/// │  (Version, Footer Length,  │
/// │       Magic Bytes)         │
/// └────────────────────────────┘
/// ```
#[derive(Debug)]
pub struct LayoutDescriptor {
    /// The absolute byte offset representing the start of the schema within the file.
    schema_offset: u64,
    /// The absolute byte offset representing the start of the layout within the file.
    layout_offset: u64,
    /// The bytes from the initial read of the file, which is assumed (for now) to be sufficiently
    /// large to contain the schema and layout.
    initial_read: Bytes,
    /// The absolute byte offset representing the start of the initial read within the file.
    /// i.e., byte 0 within `initial_read` is byte `initial_read_offset` within the file.
    initial_read_offset: u64,
    /// The deserializer to use for reading layouts.
    layout_serde: LayoutDeserializer,
}

impl LayoutDescriptor {
    pub fn try_new(
        schema_offset: u64,
        layout_offset: u64,
        initial_read: Bytes,
        initial_read_offset: u64,
        layout_serde: LayoutDeserializer,
    ) -> VortexResult<Self> {
        if initial_read_offset > schema_offset {
            vortex_bail!(
                "Schema, layout, & footer must be in the initial read, got schema at {} and initial read from {}",
                schema_offset,
                initial_read_offset,
            )
        }
        // must be enough to contain the footer, eof, and two empty fbs tables (schema & layout)
        if initial_read.len() < FOOTER_FBS_SIZE + EOF_SIZE + 2 * FBS_TABLE_PREFIX_LENGTH {
            vortex_bail!(
                "Initial read must be at least {} bytes, got {}",
                FOOTER_FBS_SIZE + EOF_SIZE + 2 * FBS_TABLE_PREFIX_LENGTH,
                initial_read.len(),
            )
        }
        // TODO(wmanning): We can make the dtype optional by providing it externally
        if schema_offset >= layout_offset {
            vortex_bail!(
                "Schema must come before the footer, got schema at {} and footer at {}",
                schema_offset,
                layout_offset,
            )
        }
        Ok(Self {
            schema_offset,
            layout_offset,
            initial_read,
            initial_read_offset,
            layout_serde,
        })
    }

    /// The start offset of the schema within the byte buffer produced by the initial read.
    fn schema_offset_relative(&self) -> usize {
        (self.schema_offset - self.initial_read_offset) as usize
    }

    /// The start offset of the layout within the byte buffer produced by the initial read.
    fn layout_offset_relative(&self) -> usize {
        (self.layout_offset - self.initial_read_offset) as usize
    }

    /// The bytes of the `Layout` flatbuffer, skipping the flatbuffer length prefix.
    fn layout_bytes(&self) -> Bytes {
        let start_offset = self.layout_offset_relative() + FBS_TABLE_PREFIX_LENGTH;
        let end_offset = self.initial_read.len() - FOOTER_FBS_SIZE - EOF_SIZE;
        self.initial_read.slice(start_offset..end_offset)
    }

    /// The bytes of the `Schema` flatbuffer, skipping the flatbuffer length prefix.
    /// Currently, the `Schema` flatbuffer only contains a DType.
    fn schema_bytes(&self) -> Bytes {
        let start_offset = self.schema_offset_relative() + FBS_TABLE_PREFIX_LENGTH;
        let end_offset = self.layout_offset_relative();
        self.initial_read.slice(start_offset..end_offset)
    }

    /// The total number of rows contained in this file.
    pub fn row_count(&self) -> VortexResult<u64> {
        let layout_bytes = self.layout_bytes();
        let fb_layout = unsafe { root_unchecked::<footer::Layout>(&layout_bytes) };
        Ok(fb_layout.row_count())
    }

    /// A [LayoutReader] which will read and produce one or more arrays from the data and metadata.
    pub fn layout_reader(
        &self,
        scan: Scan,
        message_cache: RelativeLayoutCache,
    ) -> VortexResult<Box<dyn LayoutReader>> {
        let layout_bytes = self.layout_bytes();
        let fb_layout = unsafe { root_unchecked::<footer::Layout>(&layout_bytes) };
        let fb_loc = fb_layout._tab.loc();
        self.layout_serde
            .read_layout(layout_bytes, fb_loc, scan, message_cache)
    }

    /// The (eagerly deserialized) DType from the top-level schema.
    pub fn dtype(&self) -> VortexResult<DType> {
        DType::try_from(
            self.fb_schema()?
                .dtype()
                .ok_or_else(|| vortex_err!(InvalidSerde: "Schema missing DType"))?,
        )
    }

    /// The (eagerly deserialized) DType from the top-level schema, projected to the given fields.
    pub fn projected_dtype(&self, projection: &[Field]) -> VortexResult<DType> {
        let fb_dtype = self
            .fb_schema()?
            .dtype()
            .ok_or_else(|| vortex_err!(InvalidSerde: "Schema missing DType"))?;
        deserialize_and_project(fb_dtype, projection)
    }

    /// A [LazyDeserializedDType] for the top-level schema, with the given projection applied.
    pub fn lazily_projected_dtype(
        &self,
        projection: Projection,
    ) -> VortexResult<LazilyDeserializedDType> {
        Ok(LazilyDeserializedDType::from_schema_bytes(self.schema_bytes(), projection))
    }

    /// The Footer flatbuffer.
    fn fb_footer(&self) -> VortexResult<footer::Footer> {
        let start_offset = self.layout_offset_relative() + FBS_TABLE_PREFIX_LENGTH;
        let end_offset = self.initial_read.len() - FOOTER_FBS_SIZE - EOF_SIZE;
        let footer_bytes = &self.initial_read[start_offset..end_offset];
        Ok(root::<footer::Footer>(footer_bytes)?)
    }

    fn fb_schema(&self) -> VortexResult<fb::Schema> {
        let start_offset = self.schema_offset_relative() + FBS_TABLE_PREFIX_LENGTH;
        let end_offset = self.layout_offset_relative();
        let dtype_bytes = &self.initial_read[start_offset..end_offset];

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

        let ps = root::<footer::Postscript>(&buf[eof_loc - FOOTER_FBS_SIZE..eof_loc])?;

        Ok(LayoutDescriptor {
            schema_offset: ps.schema_offset(),
            layout_offset: ps.footer_offset(),
            initial_read: buf.freeze(),
            initial_read_offset: read_offset,
            layout_serde: self.layout_serde.clone(),
        })
    }
}
