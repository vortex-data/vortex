use std::collections::BTreeSet;
use std::sync::Arc;

use bytes::Bytes;
use flatbuffers::root;
use once_cell::sync::OnceCell;
use vortex_error::{vortex_bail, VortexResult};
use vortex_flatbuffers::{footer, message};
use vortex_ipc::stream_writer::ByteRange;

use crate::read::cache::{LazyDType, RelativeLayoutCache};
use crate::read::mask::RowMask;
use crate::{
    BatchRead, Layout, LayoutDeserializer, LayoutId, LayoutPartId, LayoutReader, MessageLocator,
    Scan, INLINE_SCHEMA_LAYOUT_ID,
};

#[derive(Debug)]
pub struct InlineDTypeLayout;

impl Layout for InlineDTypeLayout {
    fn id(&self) -> LayoutId {
        INLINE_SCHEMA_LAYOUT_ID
    }

    fn reader(
        &self,
        fb_bytes: Bytes,
        fb_loc: usize,
        scan: Scan,
        layout_reader: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> VortexResult<Box<dyn LayoutReader>> {
        Ok(Box::new(InlineDTypeLayoutReader::new(
            fb_bytes,
            fb_loc,
            scan,
            layout_reader,
            message_cache,
        )))
    }
}

/// Layout that contains its own DType.
#[derive(Debug)]
pub struct InlineDTypeLayoutReader {
    fb_bytes: Bytes,
    fb_loc: usize,
    scan: Scan,
    layout_builder: LayoutDeserializer,
    message_cache: RelativeLayoutCache,
    child_layout: OnceCell<Box<dyn LayoutReader>>,
}

const INLINE_DTYPE_BUFFER_IDX: LayoutPartId = 0;
const INLINE_DTYPE_CHILD_IDX: LayoutPartId = 1;

impl InlineDTypeLayoutReader {
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
            child_layout: OnceCell::new(),
        }
    }

    fn flatbuffer(&self) -> footer::Layout {
        unsafe {
            let tab = flatbuffers::Table::new(&self.fb_bytes, self.fb_loc);
            footer::Layout::init_from_table(tab)
        }
    }

    fn dtype_message(&self) -> VortexResult<MessageLocator> {
        let buffers = self.flatbuffer().buffers().unwrap_or_default();
        if buffers.is_empty() {
            vortex_bail!("Missing buffers for inline dtype layout")
        }
        let dtype_buf = buffers.get(0);
        Ok(MessageLocator(
            self.message_cache.absolute_id(&[INLINE_DTYPE_BUFFER_IDX]),
            ByteRange::new(dtype_buf.begin(), dtype_buf.end()),
        ))
    }

    fn dtype(&self) -> VortexResult<Arc<LazyDType>> {
        if let Some(dt_bytes) = self.message_cache.get(&[INLINE_DTYPE_BUFFER_IDX]) {
            root::<message::Schema>(&dt_bytes)?;
            Ok(Arc::new(unsafe { LazyDType::from_schema_bytes(dt_bytes) }))
        } else {
            Ok(Arc::new(LazyDType::unknown()))
        }
    }

    fn child_reader(&self) -> VortexResult<Box<dyn LayoutReader>> {
        self.layout_builder.read_layout(
            self.fb_bytes.clone(),
            self.child_layout()?._tab.loc(),
            self.scan.clone(),
            self.message_cache
                .relative(INLINE_DTYPE_CHILD_IDX, self.dtype()?),
        )
    }

    fn child_layout(&self) -> VortexResult<footer::Layout> {
        let children = self.flatbuffer().children().unwrap_or_default();
        if children.is_empty() {
            vortex_bail!("Missing children for inline dtype layout")
        }
        Ok(children.get(0))
    }
}

impl LayoutReader for InlineDTypeLayoutReader {
    fn add_splits(&self, row_offset: usize, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
        self.child_reader()?.add_splits(row_offset, splits)
    }

    fn read_selection(&self, selector: &RowMask) -> VortexResult<Option<BatchRead>> {
        if let Some(cr) = self.child_layout.get() {
            cr.read_selection(selector)
        } else {
            if self.message_cache.get(&[INLINE_DTYPE_BUFFER_IDX]).is_some() {
                self.child_layout.get_or_try_init(|| self.child_reader())?;
                return self.read_selection(selector);
            }
            Ok(Some(BatchRead::ReadMore(vec![self.dtype_message()?])))
        }
    }
}
