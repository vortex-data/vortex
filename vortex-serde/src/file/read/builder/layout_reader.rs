use std::ops::Range;

use bytes::{Bytes, BytesMut};
use flatbuffers::{root, root_unchecked};
use vortex_dtype::field::Field;
use vortex_dtype::flatbuffers::deserialize_and_project;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};
use vortex_flatbuffers::{footer, message as fb};
use vortex_schema::projection::Projection;

use crate::io::VortexReadAt;
use crate::file::read::cache::{LazilyDeserializedDType, RelativeLayoutCache};
use crate::file::read::context::LayoutDeserializer;
use crate::file::read::{LayoutReader, Scan, INITIAL_READ_SIZE};
use crate::file::{EOF_SIZE, MAGIC_BYTES, V1_FOOTER_FBS_SIZE, VERSION};
use crate::MESSAGE_PREFIX_LENGTH;


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

    fn fb_schema_relative_byte_range(&self) -> Range<usize> {
        // HACK: we wrap the schema in a message right now, so we need to skip the 4-byte message prefix
        let start_offset = self.schema_offset_relative() + MESSAGE_PREFIX_LENGTH;
        let end_offset = self.layout_offset_relative();
        start_offset..end_offset
    }

    fn fb_layout_relative_byte_range(&self) -> Range<usize> {
        // HACK: we wrap the layout in a message right now, so we need to skip the 4-byte message prefix
        let start_offset = self.layout_offset_relative() + MESSAGE_PREFIX_LENGTH;
        let end_offset = self.initial_read.len() - V1_FOOTER_FBS_SIZE - EOF_SIZE;
        start_offset..end_offset
    }

    /// The start offset of the layout within the byte buffer produced by the initial read.
    fn layout_offset_relative(&self) -> usize {
        (self.layout_offset - self.initial_read_offset) as usize
    }

    /// The bytes of the `Layout` flatbuffer.
    fn layout_bytes(&self) -> Bytes {
        self.initial_read
            .slice(self.fb_layout_relative_byte_range())
    }

    /// The bytes of the `Schema` flatbuffer. Currently, the `Schema` flatbuffer only contains
    /// a single `DType` field.
    fn schema_bytes(&self) -> Bytes {
        self.initial_read
            .slice(self.fb_schema_relative_byte_range())
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
        Ok(LazilyDeserializedDType::from_schema_bytes(
            self.schema_bytes(),
            projection,
        ))
    }

    /// The `Schema` flatbuffer, which contains the top-level DType.
    fn fb_schema(&self) -> VortexResult<fb::Schema> {
        // hack: we wrap the schema in a message right now, so we need to skip the 4-byte message prefix
        let schema_msg_bytes = &self.initial_read[self.fb_schema_relative_byte_range()];
        root::<fb::Message>(schema_msg_bytes)
            .map_err(|e| e.into())
            .and_then(|m| {
                m.header_as_schema()
                    .ok_or_else(|| vortex_err!("Message was not a schema"))
            })
    }
}
