use std::cmp::min;
use std::sync::Arc;

use bytes::Bytes;
use vortex::compute::slice;
use vortex::{Array, Context};
use vortex_error::{vortex_err, VortexResult, VortexUnwrap};
use vortex_flatbuffers::footer;

use crate::layouts::read::cache::RelativeLayoutCache;
use crate::layouts::{
    LayoutDeserializer, LayoutId, LayoutReader, LayoutSpec, Message, ReadResult, Scan,
    FLAT_LAYOUT_ID,
};
use crate::message_reader::ArrayBufferReader;
use crate::stream_writer::ByteRange;

#[derive(Debug)]
pub struct FlatLayoutSpec;

impl LayoutSpec for FlatLayoutSpec {
    fn id(&self) -> LayoutId {
        FLAT_LAYOUT_ID
    }

    fn layout(
        &self,
        fb_bytes: Bytes,
        fb_loc: usize,
        scan: Scan,
        layout_serde: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> Box<dyn LayoutReader> {
        let fb_layout = unsafe {
            let tab = flatbuffers::Table::new(&fb_bytes, fb_loc);
            footer::Layout::init_from_table(tab)
        };
        let buf = fb_layout
            .buffers()
            .ok_or_else(|| vortex_err!("No buffers"))
            .vortex_unwrap()
            .get(0);

        Box::new(FlatLayout::new(
            ByteRange::new(buf.begin(), buf.end()),
            scan,
            layout_serde.ctx(),
            message_cache,
        ))
    }
}

#[derive(Debug)]
pub struct FlatLayout {
    range: ByteRange,
    scan: Scan,
    ctx: Arc<Context>,
    cache: RelativeLayoutCache,
    done: bool,
    cached_array: Option<Array>,
}

impl FlatLayout {
    pub fn new(
        range: ByteRange,
        scan: Scan,
        ctx: Arc<Context>,
        cache: RelativeLayoutCache,
    ) -> Self {
        Self {
            range,
            scan,
            ctx,
            cache,
            done: false,
            cached_array: None,
        }
    }

    fn own_message(&self) -> Message {
        (self.cache.absolute_id(&[]), self.range)
    }

    fn array_from_bytes(&self, mut buf: Bytes) -> VortexResult<Array> {
        let mut array_reader = ArrayBufferReader::new();
        let mut read_buf = Bytes::new();
        while let Some(u) = array_reader.read(read_buf)? {
            read_buf = buf.split_to(u);
        }
        array_reader.into_array(self.ctx.clone(), self.cache.dtype().value()?.clone())
    }
}

impl LayoutReader for FlatLayout {
    fn read_next(&mut self) -> VortexResult<Option<ReadResult>> {
        if self.done {
            return Ok(None);
        }

        if let Some(array) = self.cached_array.take() {
            let array = if array.len() > self.scan.batch_size {
                let rows_to_read = min(self.scan.batch_size, array.len());
                let taken = slice(&array, 0, rows_to_read)?;
                let leftover = slice(&array, rows_to_read, array.len())?;
                self.cached_array = Some(leftover);
                taken
            } else {
                self.done = true;
                array
            };
            Ok(Some(ReadResult::Batch(array)))
        } else if let Some(buf) = self.cache.get(&[]) {
            self.cached_array = Some(self.array_from_bytes(buf)?);
            self.read_next()
        } else {
            Ok(Some(ReadResult::ReadMore(vec![self.own_message()])))
        }
    }
}
