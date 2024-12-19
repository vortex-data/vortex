use std::collections::BTreeSet;
use std::io::Cursor;
use std::sync::Arc;

use bytes::Bytes;
use vortex_array::{ArrayData, Context};
use vortex_error::{vortex_bail, VortexResult};
use vortex_flatbuffers::footer;
use vortex_ipc::messages::{DecoderMessage, SyncMessageReader};

use crate::byte_range::ByteRange;
use crate::read::cache::RelativeLayoutCache;
use crate::read::mask::RowMask;
use crate::{
    Layout, LayoutDeserializer, LayoutId, LayoutReader, MessageLocator, PollRead, Scan,
    FLAT_LAYOUT_ID,
};

#[derive(Debug)]
pub struct FlatLayout;

impl Layout for FlatLayout {
    fn id(&self) -> LayoutId {
        FLAT_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: footer::Layout,
        scan: Scan,
        layout_serde: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> VortexResult<Box<dyn LayoutReader>> {
        let buffers = layout.buffers().unwrap_or_default();
        if buffers.len() != 1 {
            vortex_bail!("Flat layout can have exactly 1 buffer")
        }
        let buf = buffers.get(0);

        Ok(Box::new(FlatLayoutReader::new(
            ByteRange::new(buf.begin(), buf.end()),
            scan,
            layout_serde.ctx(),
            message_cache,
        )))
    }
}

#[derive(Debug)]
pub struct FlatLayoutReader {
    range: ByteRange,
    scan: Scan,
    ctx: Arc<Context>,
    message_cache: RelativeLayoutCache,
}

impl FlatLayoutReader {
    pub fn new(
        range: ByteRange,
        scan: Scan,
        ctx: Arc<Context>,
        message_cache: RelativeLayoutCache,
    ) -> Self {
        Self {
            range,
            scan,
            ctx,
            message_cache,
        }
    }

    fn own_message(&self) -> MessageLocator {
        MessageLocator(self.message_cache.absolute_id(&[]), self.range)
    }

    fn array_from_bytes(&self, buf: Bytes) -> VortexResult<ArrayData> {
        let mut reader = SyncMessageReader::new(Cursor::new(buf));
        match reader.next().transpose()? {
            Some(DecoderMessage::Array(array_parts)) => array_parts.into_array_data(
                self.ctx.clone(),
                self.message_cache.dtype().value()?.clone(),
            ),
            Some(msg) => vortex_bail!("Expected Array message, got {:?}", msg),
            None => vortex_bail!("Expected Array message, got EOF"),
        }
    }
}

impl LayoutReader for FlatLayoutReader {
    fn add_splits(&self, row_offset: usize, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
        splits.insert(row_offset);
        Ok(())
    }

    fn poll_read(&self, selection: &RowMask) -> VortexResult<Option<PollRead<ArrayData>>> {
        if let Some(buf) = self.message_cache.get(&[]) {
            let array = self.array_from_bytes(buf)?;
            selection
                .filter_array(array)?
                .map(|s| {
                    Ok(PollRead::Value(
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
            Ok(Some(PollRead::ReadMore(vec![self.own_message()])))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use bytes::Bytes;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::{Context, IntoArrayData, IntoArrayVariant, ToArrayData};
    use vortex_dtype::PType;
    use vortex_expr::{BinaryExpr, Identity, Literal, Operator};
    use vortex_ipc::messages::{EncoderMessage, SyncMessageWriter};

    use crate::byte_range::ByteRange;
    use crate::layouts::flat::FlatLayoutReader;
    use crate::read::cache::{LazyDType, RelativeLayoutCache};
    use crate::read::layouts::test_read::{filter_read_layout, read_layout};
    use crate::{LayoutMessageCache, RowFilter, Scan};

    async fn read_only_layout(
        cache: Arc<RwLock<LayoutMessageCache>>,
    ) -> (FlatLayoutReader, Bytes, usize, Arc<LazyDType>) {
        let array = PrimitiveArray::from((0..100).collect::<Vec<_>>()).into_array();

        let mut written = vec![];
        SyncMessageWriter::new(&mut written)
            .write_message(EncoderMessage::Array(&array.to_array()))
            .unwrap();

        let projection_scan = Scan::empty();
        let dtype = Arc::new(LazyDType::from_dtype(PType::I32.into()));

        (
            FlatLayoutReader::new(
                ByteRange::new(0, written.len() as u64),
                projection_scan,
                Arc::new(Context::default()),
                RelativeLayoutCache::new(cache, dtype.clone()),
            ),
            Bytes::from(written),
            array.len(),
            dtype,
        )
    }

    async fn layout_and_bytes(
        cache: Arc<RwLock<LayoutMessageCache>>,
        scan: Scan,
    ) -> (FlatLayoutReader, FlatLayoutReader, Bytes, usize) {
        let (read_layout, bytes, len, dtype) = read_only_layout(cache.clone()).await;

        (
            FlatLayoutReader::new(
                ByteRange::new(0, bytes.len() as u64),
                scan,
                Arc::new(Context::default()),
                RelativeLayoutCache::new(cache, dtype),
            ),
            read_layout,
            bytes,
            len,
        )
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_range() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut filter_layout, mut projection_layout, buf, length) = layout_and_bytes(
            cache.clone(),
            Scan::new(RowFilter::new_expr(BinaryExpr::new_expr(
                Arc::new(Identity),
                Operator::Gt,
                Literal::new_expr(10.into()),
            ))),
        )
        .await;
        let arr = filter_read_layout(
            &mut filter_layout,
            &mut projection_layout,
            cache,
            &buf,
            length,
        )
        .pop_front();

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
        let (mut data_layout, buf, length, ..) = read_only_layout(cache.clone()).await;
        let arr = read_layout(&mut data_layout, cache, &buf, length).pop_front();

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
        let (mut filter_layout, mut projection_layout, buf, length) = layout_and_bytes(
            cache.clone(),
            Scan::new(RowFilter::new_expr(BinaryExpr::new_expr(
                Arc::new(Identity),
                Operator::Gt,
                Literal::new_expr(101.into()),
            ))),
        )
        .await;
        let arr = filter_read_layout(
            &mut filter_layout,
            &mut projection_layout,
            cache,
            &buf,
            length,
        )
        .pop_front();

        assert!(arr.is_none());
    }
}
