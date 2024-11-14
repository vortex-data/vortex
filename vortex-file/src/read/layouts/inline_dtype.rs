use std::collections::BTreeSet;
use std::sync::Arc;

use bytes::Bytes;
use flatbuffers::root;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_flatbuffers::{footer, message};
use vortex_ipc::messages::reader::MESSAGE_PREFIX_LENGTH;
use vortex_ipc::stream_writer::ByteRange;

use crate::read::cache::{LazilyDeserializedDType, RelativeLayoutCache};
use crate::read::mask::RowMask;
use crate::{
    BatchRead, LayoutDeserializer, LayoutId, LayoutPartId, LayoutReader, LayoutSpec,
    MessageLocator, Scan, INLINE_SCHEMA_LAYOUT_ID,
};

#[derive(Debug)]
pub struct InlineDTypeLayoutSpec;

impl LayoutSpec for InlineDTypeLayoutSpec {
    fn id(&self) -> LayoutId {
        INLINE_SCHEMA_LAYOUT_ID
    }

    fn layout_reader(
        &self,
        fb_bytes: Bytes,
        fb_loc: usize,
        scan: Scan,
        layout_reader: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> VortexResult<Box<dyn LayoutReader>> {
        Ok(Box::new(InlineDTypeLayout::new(
            fb_bytes,
            fb_loc,
            scan,
            layout_reader,
            message_cache,
        )))
    }
}

#[derive(Debug)]
pub struct InlineDTypeLayout {
    fb_bytes: Bytes,
    fb_loc: usize,
    scan: Scan,
    layout_builder: LayoutDeserializer,
    message_cache: RelativeLayoutCache,
    child_layout: Option<Box<dyn LayoutReader>>,
}

enum DTypeReadResult {
    ReadMore(Vec<MessageLocator>),
    DType(DType),
}

enum ChildReaderResult {
    ReadMore(Vec<MessageLocator>),
    Reader(Box<dyn LayoutReader>),
}

const INLINE_DTYPE_BUFFER_IDX: LayoutPartId = 0;
const INLINE_DTYPE_CHILD_IDX: LayoutPartId = 1;

impl InlineDTypeLayout {
    pub fn new(
        fb_bytes: Bytes,
        fb_loc: usize,
        scan: Scan,
        layout_builder: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> Self {
        Self {
            fb_bytes,
            fb_loc,
            scan,
            layout_builder,
            message_cache,
            child_layout: None,
        }
    }

    fn flatbuffer(&self) -> footer::Layout {
        unsafe {
            let tab = flatbuffers::Table::new(&self.fb_bytes, self.fb_loc);
            footer::Layout::init_from_table(tab)
        }
    }

    fn dtype(&self) -> VortexResult<DTypeReadResult> {
        if let Some(dt_bytes) = self.message_cache.get(&[INLINE_DTYPE_BUFFER_IDX]) {
            let msg = root::<message::Message>(&dt_bytes[MESSAGE_PREFIX_LENGTH..])?
                .header_as_schema()
                .ok_or_else(|| vortex_err!("Expected schema message"))?;

            Ok(DTypeReadResult::DType(DType::try_from(
                msg.dtype()
                    .ok_or_else(|| vortex_err!(InvalidSerde: "Schema missing DType"))?,
            )?))
        } else {
            let buffers = self.flatbuffer().buffers().unwrap_or_default();
            if buffers.is_empty() {
                vortex_bail!("Missing buffers for inline dtype layout")
            }
            let dtype_buf = buffers.get(0);
            Ok(DTypeReadResult::ReadMore(vec![MessageLocator(
                self.message_cache.absolute_id(&[INLINE_DTYPE_BUFFER_IDX]),
                ByteRange::new(dtype_buf.begin(), dtype_buf.end()),
            )]))
        }
    }

    fn child_reader(&self) -> VortexResult<ChildReaderResult> {
        match self.dtype()? {
            DTypeReadResult::ReadMore(m) => Ok(ChildReaderResult::ReadMore(m)),
            DTypeReadResult::DType(d) => {
                let child_layout = self.layout_builder.read_layout(
                    self.fb_bytes.clone(),
                    self.child_layout()?._tab.loc(),
                    self.scan.clone(),
                    self.message_cache.relative(
                        INLINE_DTYPE_CHILD_IDX,
                        Arc::new(LazilyDeserializedDType::from_dtype(d)),
                    ),
                )?;
                Ok(ChildReaderResult::Reader(child_layout))
            }
        }
    }

    fn child_layout(&self) -> VortexResult<footer::Layout> {
        let children = self.flatbuffer().children().unwrap_or_default();
        if children.is_empty() {
            vortex_bail!("Missing children for inline dtype layout")
        }
        Ok(children.get(0))
    }
}

impl LayoutReader for InlineDTypeLayout {
    fn add_splits(&self, row_offset: usize, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
        let child_layout = self.layout_builder.read_layout(
            self.fb_bytes.clone(),
            self.child_layout()?._tab.loc(),
            Scan::new(None),
            self.message_cache.unknown_dtype(INLINE_DTYPE_CHILD_IDX),
        )?;
        child_layout.add_splits(row_offset, splits)
    }

    fn read_selection(&mut self, selector: &RowMask) -> VortexResult<Option<BatchRead>> {
        if let Some(cr) = self.child_layout.as_mut() {
            cr.read_selection(selector)
        } else {
            match self.child_reader()? {
                ChildReaderResult::ReadMore(rm) => Ok(Some(BatchRead::ReadMore(rm))),
                ChildReaderResult::Reader(r) => {
                    self.child_layout = Some(r);
                    self.read_selection(selector)
                }
            }
        }
    }
}
