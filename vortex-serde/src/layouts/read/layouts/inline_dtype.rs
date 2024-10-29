use std::sync::Arc;

use bytes::Bytes;
use flatbuffers::root;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult};
use vortex_flatbuffers::{footer, message};

use crate::layouts::read::cache::{LazyDeserializedDType, RelativeLayoutCache};
use crate::layouts::read::selection::RowSelector;
use crate::layouts::{
    LayoutDeserializer, LayoutId, LayoutReader, LayoutSpec, Message, RangeResult, ReadResult, Scan,
    INLINE_SCHEMA_LAYOUT_ID,
};
use crate::message_reader::FLATBUFFER_SIZE_LENGTH;
use crate::stream_writer::ByteRange;

#[derive(Debug)]
pub struct InlineDTypeLayoutSpec;

impl LayoutSpec for InlineDTypeLayoutSpec {
    fn id(&self) -> LayoutId {
        INLINE_SCHEMA_LAYOUT_ID
    }

    fn layout(
        &self,
        fb_bytes: Bytes,
        fb_loc: usize,
        scan: Scan,
        layout_reader: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> Box<dyn LayoutReader> {
        Box::new(InlineDTypeLayout::new(
            fb_bytes,
            fb_loc,
            scan,
            layout_reader,
            message_cache,
        ))
    }
}

#[derive(Debug)]
pub struct InlineDTypeLayout {
    fb_bytes: Bytes,
    fb_loc: usize,
    offset: usize,
    scan: Scan,
    layout_builder: LayoutDeserializer,
    message_cache: RelativeLayoutCache,
    child_layout: Option<Box<dyn LayoutReader>>,
}

enum DTypeReadResult {
    ReadMore(Vec<Message>),
    DType(DType),
}

enum ChildReaderResult {
    ReadMore(Vec<Message>),
    Reader(Box<dyn LayoutReader>),
}

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
            offset: 0,
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
        if let Some(dt_bytes) = self.message_cache.get(&[0]) {
            let msg = root::<message::Message>(&dt_bytes[FLATBUFFER_SIZE_LENGTH..])?
                .header_as_schema()
                .ok_or_else(|| vortex_err!("Expected schema message"))?;

            Ok(DTypeReadResult::DType(
                DType::try_from(
                    msg.dtype()
                        .ok_or_else(|| vortex_err!(InvalidSerde: "Schema missing DType"))?,
                )
                .map_err(|e| vortex_err!(InvalidSerde: "Failed to parse DType: {e}"))?,
            ))
        } else {
            let dtype_buf = self
                .flatbuffer()
                .buffers()
                .ok_or_else(|| vortex_err!("No buffers"))?
                .get(0);
            Ok(DTypeReadResult::ReadMore(vec![(
                self.message_cache.absolute_id(&[0]),
                ByteRange::new(dtype_buf.begin(), dtype_buf.end()),
            )]))
        }
    }

    fn child_reader(&mut self) -> VortexResult<ChildReaderResult> {
        match self.dtype()? {
            DTypeReadResult::ReadMore(m) => Ok(ChildReaderResult::ReadMore(m)),
            DTypeReadResult::DType(d) => {
                let layout = self
                    .flatbuffer()
                    .children()
                    .ok_or_else(|| vortex_err!("No children"))?
                    .get(0);

                let mut child_layout = self.layout_builder.read_layout(
                    self.fb_bytes.clone(),
                    layout._tab.loc(),
                    self.scan.clone(),
                    self.message_cache
                        .relative(1u16, Arc::new(LazyDeserializedDType::from_dtype(d))),
                )?;
                if self.offset != 0 {
                    child_layout.advance(self.offset)?;
                }
                Ok(ChildReaderResult::Reader(child_layout))
            }
        }
    }
}

impl LayoutReader for InlineDTypeLayout {
    fn next_range(&mut self) -> VortexResult<RangeResult> {
        if let Some(cr) = self.child_layout.as_mut() {
            cr.next_range()
        } else {
            match self.child_reader()? {
                ChildReaderResult::ReadMore(rm) => Ok(RangeResult::ReadMore(rm)),
                ChildReaderResult::Reader(r) => {
                    self.child_layout = Some(r);
                    self.next_range()
                }
            }
        }
    }

    fn read_next(&mut self, selector: RowSelector) -> VortexResult<Option<ReadResult>> {
        if let Some(cr) = self.child_layout.as_mut() {
            cr.read_next(selector)
        } else {
            match self.child_reader()? {
                ChildReaderResult::ReadMore(rm) => Ok(Some(ReadResult::ReadMore(rm))),
                ChildReaderResult::Reader(r) => {
                    self.child_layout = Some(r);
                    self.read_next(selector)
                }
            }
        }
    }

    fn advance(&mut self, up_to_row: usize) -> VortexResult<Vec<Message>> {
        if let Some(cr) = self.child_layout.as_mut() {
            cr.advance(up_to_row)
        } else {
            self.offset = up_to_row;
            Ok(vec![])
        }
    }
}
