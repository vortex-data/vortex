use std::sync::Arc;

use bytes::Bytes;
use croaring::Bitmap;
use vortex::compute::slice;
use vortex::{Array, Context};
use vortex_error::{vortex_bail, vortex_err, VortexResult, VortexUnwrap};
use vortex_flatbuffers::footer;

use crate::layouts::read::cache::RelativeLayoutCache;
use crate::layouts::read::selection::RowSelector;
use crate::layouts::{
    LayoutDeserializer, LayoutId, LayoutReader, LayoutSpec, Message, RangeResult, ReadResult, Scan,
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
        length: u64,
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
            length,
            scan,
            layout_serde.ctx(),
            message_cache,
        ))
    }
}

#[derive(Debug)]
pub struct FlatLayout {
    range: ByteRange,
    length: u64,
    scan: Scan,
    ctx: Arc<Context>,
    cache: RelativeLayoutCache,
    offset: usize,
    sent_range: bool,
}

impl FlatLayout {
    pub fn new(
        range: ByteRange,
        length: u64,
        scan: Scan,
        ctx: Arc<Context>,
        cache: RelativeLayoutCache,
    ) -> Self {
        Self {
            range,
            length,
            scan,
            ctx,
            cache,
            offset: 0,
            sent_range: false,
        }
    }

    fn skipped(&self) -> bool {
        self.offset as u64 == self.length
    }

    fn own_range(&self) -> Option<RowSelector> {
        (self.offset as u64 != self.length).then(|| {
            RowSelector::new(
                Bitmap::from_range(self.offset as u32..self.length as u32),
                self.offset,
                self.length as usize,
            )
        })
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
    fn next_range(&mut self) -> VortexResult<RangeResult> {
        if self.sent_range {
            Ok(RangeResult::Rows(None))
        } else {
            self.sent_range = true;
            Ok(RangeResult::Rows(self.own_range()))
        }
    }

    fn read_next(&mut self, selection: RowSelector) -> VortexResult<Option<ReadResult>> {
        if self.skipped() || selection.end() <= self.offset {
            return Ok(None);
        }

        if let Some(buf) = self.cache.get(&[]) {
            let array = self.array_from_bytes(buf)?;
            let selection_end = selection.end();
            let selected = selection.offset(self.offset as i64).filter_array(slice(
                &array,
                self.offset,
                selection_end,
            )?)?;
            self.offset = selection_end;
            selected
                .map(|s| {
                    Ok(ReadResult::Batch(
                        self.scan
                            .expr
                            .as_ref()
                            .map(|e| e.evaluate(&s))
                            .transpose()?
                            .unwrap_or(s),
                    ))
                })
                .transpose()
        } else {
            Ok(Some(ReadResult::ReadMore(vec![self.own_message()])))
        }
    }

    fn advance(&mut self, up_to_row: usize) -> VortexResult<Vec<Message>> {
        if up_to_row < self.offset {
            vortex_bail!("Can't advance backwards")
        }

        self.offset = up_to_row;
        if self.skipped() {
            Ok(vec![])
        } else {
            Ok(vec![self.own_message()])
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, RwLock};

    use bytes::Bytes;
    use croaring::Bitmap;
    use vortex::array::PrimitiveArray;
    use vortex::{Array, Context, IntoArray, IntoArrayVariant};
    use vortex_dtype::PType;
    use vortex_expr::{BinaryExpr, Identity, Literal, Operator};

    use crate::layouts::read::cache::{LazyDeserializedDType, RelativeLayoutCache};
    use crate::layouts::read::layouts::flat::FlatLayout;
    use crate::layouts::read::layouts::test_read::{
        filter_read_layout, read_filters, read_layout, read_layout_data, read_layout_ranges,
    };
    use crate::layouts::read::selection::RowSelector;
    use crate::layouts::{LayoutMessageCache, LayoutReader, RowFilter, Scan};
    use crate::message_writer::MessageWriter;
    use crate::stream_writer::ByteRange;

    async fn read_only_layout(
        cache: Arc<RwLock<LayoutMessageCache>>,
    ) -> (FlatLayout, Bytes, u64, Arc<LazyDeserializedDType>) {
        let mut writer = MessageWriter::new(Vec::new());
        let array = PrimitiveArray::from((0..100).collect::<Vec<_>>()).into_array();
        let len = array.len() as u64;
        writer.write_batch(array).await.unwrap();
        let written = writer.into_inner();

        let projection_scan = Scan::new(None);
        let dtype = Arc::new(LazyDeserializedDType::from_dtype(PType::I32.into()));

        (
            FlatLayout::new(
                ByteRange::new(0, written.len() as u64),
                len,
                projection_scan,
                Arc::new(Context::default()),
                RelativeLayoutCache::new(cache, dtype.clone()),
            ),
            Bytes::from(written),
            len,
            dtype,
        )
    }

    async fn layout_and_bytes(
        cache: Arc<RwLock<LayoutMessageCache>>,
        scan: Scan,
    ) -> (FlatLayout, FlatLayout, Bytes) {
        let (read_layout, bytes, len, dtype) = read_only_layout(cache.clone()).await;

        (
            FlatLayout::new(
                ByteRange::new(0, bytes.len() as u64),
                len,
                scan,
                Arc::new(Context::default()),
                RelativeLayoutCache::new(cache, dtype),
            ),
            read_layout,
            bytes,
        )
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_range() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut filter_layout, mut projection_layout, buf) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(Arc::new(RowFilter::new(Arc::new(BinaryExpr::new(
                Arc::new(Identity),
                Operator::Gt,
                Arc::new(Literal::new(10.into())),
            )))))),
        )
        .await;
        let arr =
            filter_read_layout(&mut filter_layout, &mut projection_layout, cache, &buf).pop_front();

        assert!(arr.is_some());
        let arr = arr.unwrap();
        assert_eq!(
            arr.into_primitive().unwrap().maybe_null_slice::<i32>(),
            &(11..100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_range_no_filter() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut data_layout, buf, ..) = read_only_layout(cache.clone()).await;
        let arr = read_layout(&mut data_layout, cache, &buf).pop_front();

        assert!(arr.is_some());
        let arr = arr.unwrap();
        assert_eq!(
            arr.into_primitive().unwrap().maybe_null_slice::<i32>(),
            &(0..100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_empty() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut filter_layout, mut projection_layout, buf) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(Arc::new(RowFilter::new(Arc::new(BinaryExpr::new(
                Arc::new(Identity),
                Operator::Gt,
                Arc::new(Literal::new(101.into())),
            )))))),
        )
        .await;
        let arr =
            filter_read_layout(&mut filter_layout, &mut projection_layout, cache, &buf).pop_front();

        assert!(arr.is_none());
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn advance_read_range() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut filter_layout, mut projection_layout, buf) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(Arc::new(RowFilter::new(Arc::new(BinaryExpr::new(
                Arc::new(Identity),
                Operator::Gt,
                Arc::new(Literal::new(10.into())),
            )))))),
        )
        .await;
        filter_layout.advance(50).unwrap();
        let arr =
            filter_read_layout(&mut filter_layout, &mut projection_layout, cache, &buf).pop_front();

        assert!(arr.is_some());
        let arr = arr.unwrap();
        assert_eq!(
            arr.into_primitive().unwrap().maybe_null_slice::<i32>(),
            &(50..100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn advance_skipped() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut filter_layout, mut projection_layout, buf) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(Arc::new(RowFilter::new(Arc::new(BinaryExpr::new(
                Arc::new(Identity),
                Operator::Gt,
                Arc::new(Literal::new(10.into())),
            )))))),
        )
        .await;
        filter_layout.advance(100).unwrap();
        let arr =
            filter_read_layout(&mut filter_layout, &mut projection_layout, cache, &buf).pop_front();

        assert!(arr.is_none());
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_multiple_selectors() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (_, mut projection_layout, buf) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(Arc::new(RowFilter::new(Arc::new(BinaryExpr::new(
                Arc::new(Identity),
                Operator::Gt,
                Arc::new(Literal::new(10.into())),
            )))))),
        )
        .await;
        let mut arr = [
            RowSelector::new(Bitmap::from_range(11..50), 0, 50),
            RowSelector::new(Bitmap::from_range(50..100), 50, 100),
        ]
        .into_iter()
        .flat_map(|s| read_layout_data(&mut projection_layout, cache.clone(), &buf, s))
        .collect::<VecDeque<_>>();

        assert_eq!(
            arr.pop_front()
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(11..50).collect::<Vec<_>>()
        );
        assert_eq!(
            arr.pop_front()
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(50..100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn advance_after_filter() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut filter_layout, mut projection_layout, buf) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(Arc::new(RowFilter::new(Arc::new(BinaryExpr::new(
                Arc::new(Identity),
                Operator::Gt,
                Arc::new(Literal::new(10.into())),
            )))))),
        )
        .await;
        let selector = read_layout_ranges(&mut filter_layout, cache.clone(), &buf)
            .into_iter()
            .flat_map(|s| read_filters(&mut filter_layout, cache.clone(), &buf, s))
            .collect::<Vec<_>>();

        projection_layout.advance(50).unwrap();
        let mut arr: Vec<Array> = selector
            .into_iter()
            .flat_map(|s| read_layout_data(&mut projection_layout, cache.clone(), &buf, s))
            .collect();

        assert_eq!(
            arr.remove(0)
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(50..100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn advance_mid_read() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (_, mut projection_layout, buf) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(Arc::new(RowFilter::new(Arc::new(BinaryExpr::new(
                Arc::new(Identity),
                Operator::Gt,
                Arc::new(Literal::new(10.into())),
            )))))),
        )
        .await;
        let advanced = AtomicBool::new(false);
        let mut arr = [
            RowSelector::new(Bitmap::from_range(11..50), 0, 50),
            RowSelector::new(Bitmap::from_range(50..100), 50, 100),
        ]
        .into_iter()
        .flat_map(|s| {
            let a = read_layout_data(&mut projection_layout, cache.clone(), &buf, s);
            if advanced
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                projection_layout.advance(90).unwrap();
            }
            a
        })
        .collect::<VecDeque<_>>();

        assert_eq!(
            arr.pop_front()
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(11..50).collect::<Vec<_>>()
        );
        assert_eq!(
            arr.pop_front()
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(90..100).collect::<Vec<_>>()
        );
    }
}
