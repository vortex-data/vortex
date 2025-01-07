use std::collections::BTreeSet;
use std::sync::Arc;

use bytes::Bytes;
use vortex_array::{ArrayData, ContextRef};
use vortex_error::{vortex_bail, VortexResult};
use vortex_flatbuffers::footer;
use vortex_ipc::messages::{BufMessageReader, DecoderMessage};

use crate::byte_range::ByteRange;
use crate::read::mask::RowMask;
use crate::{
    Layout, LayoutDeserializer, LayoutId, LayoutPath, LayoutReader, LazyDType, MessageCache,
    MessageLocator, PollRead, Scan, FLAT_LAYOUT_ID,
};

#[derive(Debug)]
pub struct FlatLayout;

impl Layout for FlatLayout {
    fn id(&self) -> LayoutId {
        FLAT_LAYOUT_ID
    }

    fn reader(
        &self,
        path: LayoutPath,
        layout: footer::Layout,
        dtype: Arc<LazyDType>,
        scan: Scan,
        layout_serde: LayoutDeserializer,
    ) -> VortexResult<Arc<dyn LayoutReader>> {
        let buffers = layout.buffers().unwrap_or_default();
        if buffers.len() != 1 {
            vortex_bail!("Flat layout can have exactly 1 buffer")
        }
        let buf = buffers.get(0);

        Ok(Arc::new(FlatLayoutReader {
            path,
            range: ByteRange::new(buf.begin(), buf.end()),
            scan,
            dtype,
            ctx: layout_serde.ctx(),
        }))
    }
}

#[derive(Debug)]
pub struct FlatLayoutReader {
    path: LayoutPath,
    range: ByteRange,
    scan: Scan,
    dtype: Arc<LazyDType>,
    ctx: ContextRef,
}

impl FlatLayoutReader {
    fn own_message(&self) -> MessageLocator {
        MessageLocator(self.path.clone(), self.range)
    }

    fn array_from_bytes(&self, buf: Bytes) -> VortexResult<ArrayData> {
        let mut reader = BufMessageReader::new(buf);
        match reader.next().transpose()? {
            Some(DecoderMessage::Array(array_parts)) => {
                array_parts.into_array_data(self.ctx.clone(), self.dtype.value()?.clone())
            }
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

    fn poll_read(
        &self,
        selection: &RowMask,
        msgs: &dyn MessageCache,
    ) -> VortexResult<Option<PollRead<ArrayData>>> {
        if let Some(buf) = msgs.get(&self.path) {
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
    use std::sync::Arc;

    use bytes::Bytes;
    use vortex_array::{Context, IntoArrayData, IntoArrayVariant, ToArrayData};
    use vortex_buffer::Buffer;
    use vortex_dtype::PType;
    use vortex_expr::{gt, lit, BinaryExpr, Identity, Operator, RowFilter};
    use vortex_ipc::messages::{EncoderMessage, SyncMessageWriter};

    use crate::byte_range::ByteRange;
    use crate::layouts::flat::FlatLayoutReader;
    use crate::read::cache::LazyDType;
    use crate::read::layouts::test_read::{filter_read_layout, read_layout};
    use crate::{LayoutMessageCache, LayoutPath, Scan};

    async fn read_only_layout() -> (FlatLayoutReader, Bytes, usize, Arc<LazyDType>) {
        let array = Buffer::from_iter(0..100).into_array();

        let mut written = vec![];
        SyncMessageWriter::new(&mut written)
            .write_message(EncoderMessage::Array(&array.to_array()))
            .unwrap();

        let projection_scan = Scan::empty();
        let dtype = Arc::new(LazyDType::from_dtype(PType::I32.into()));

        (
            FlatLayoutReader {
                path: LayoutPath::default(),
                range: ByteRange::new(0, written.len() as u64),
                scan: projection_scan,
                dtype: dtype.clone(),
                ctx: Arc::new(Context::default()),
            },
            Bytes::from(written),
            array.len(),
            dtype,
        )
    }

    async fn layout_and_bytes(scan: Scan) -> (FlatLayoutReader, FlatLayoutReader, Bytes, usize) {
        let (read_layout, bytes, len, dtype) = read_only_layout().await;

        (
            FlatLayoutReader {
                path: LayoutPath::default(),
                range: ByteRange::new(0, bytes.len() as u64),
                scan,
                ctx: Arc::new(Context::default()),
                dtype,
            },
            read_layout,
            bytes,
            len,
        )
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_range() {
        let msgs = LayoutMessageCache::default();
        let (filter_layout, projection_layout, buf, length) = layout_and_bytes(Scan::new(
            RowFilter::new_expr(gt(Arc::new(Identity), lit(10))),
        ))
        .await;
        let arr =
            filter_read_layout(&filter_layout, &projection_layout, &buf, length, msgs).pop_front();

        assert!(arr.is_some());
        let arr = arr.unwrap();
        assert_eq!(
            arr.into_primitive().unwrap().as_slice::<i32>(),
            &(11..100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_range_no_filter() {
        let msgs = LayoutMessageCache::default();
        let (data_layout, buf, length, ..) = read_only_layout().await;
        let arr = read_layout(&data_layout, &buf, length, msgs).pop_front();

        assert!(arr.is_some());
        let arr = arr.unwrap();
        assert_eq!(
            arr.into_primitive().unwrap().as_slice::<i32>(),
            &(0..100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_empty() {
        let msgs = LayoutMessageCache::default();
        let (filter_layout, projection_layout, buf, length) =
            layout_and_bytes(Scan::new(RowFilter::new_expr(BinaryExpr::new_expr(
                Arc::new(Identity),
                Operator::Gt,
                lit(101),
            ))))
            .await;
        let arr =
            filter_read_layout(&filter_layout, &projection_layout, &buf, length, msgs).pop_front();

        assert!(arr.is_none());
    }
}
