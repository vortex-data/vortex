use std::collections::{BTreeSet, VecDeque};

use bytes::Bytes;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_flatbuffers::footer;

use crate::read::buffered::{BufferedLayoutReader, RangedLayoutReader};
use crate::read::cache::RelativeLayoutCache;
use crate::read::mask::RowMask;
use crate::{
    BatchRead, LayoutDeserializer, LayoutId, LayoutPartId, LayoutReader, LayoutSpec, Scan,
    CHUNKED_LAYOUT_ID,
};
#[derive(Default, Debug)]
pub struct ChunkedLayoutSpec;

impl LayoutSpec for ChunkedLayoutSpec {
    fn id(&self) -> LayoutId {
        CHUNKED_LAYOUT_ID
    }

    fn layout_reader(
        &self,
        fb_bytes: Bytes,
        fb_loc: usize,
        scan: Scan,
        layout_builder: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> VortexResult<Box<dyn LayoutReader>> {
        Ok(Box::new(ChunkedLayout::new(
            fb_bytes,
            fb_loc,
            scan,
            layout_builder,
            message_cache,
        )))
    }
}

/// In memory representation of Chunked NestedLayout.
///
/// First child in the list is the metadata table
/// Subsequent children are consecutive chunks of this layout
#[derive(Debug)]
pub struct ChunkedLayout {
    fb_bytes: Bytes,
    fb_loc: usize,
    scan: Scan,
    layout_builder: LayoutDeserializer,
    message_cache: RelativeLayoutCache,
    chunk_reader: Option<BufferedLayoutReader>,
}

impl ChunkedLayout {
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
            chunk_reader: None,
        }
    }

    fn flatbuffer(&self) -> footer::Layout {
        unsafe {
            let tab = flatbuffers::Table::new(&self.fb_bytes, self.fb_loc);
            footer::Layout::init_from_table(tab)
        }
    }

    fn has_metadata(&self) -> bool {
        self.flatbuffer()
            .metadata()
            .map(|b| b.bytes()[0] != 0)
            .unwrap_or(false)
    }

    fn children(&self) -> impl Iterator<Item = (usize, footer::Layout)> {
        self.flatbuffer()
            .children()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .skip(if self.has_metadata() { 1 } else { 0 })
    }

    fn child_ranges(&self) -> Vec<(usize, usize)> {
        self.children()
            .map(|(_, c)| c.row_count())
            .scan(0u64, |acc, row_count| {
                let current = *acc;
                *acc += row_count;
                Some((current as usize, *acc as usize))
            })
            .collect::<Vec<_>>()
    }

    fn child_layouts<C: Fn(LayoutPartId) -> RelativeLayoutCache>(
        &self,
        cache: C,
    ) -> VortexResult<VecDeque<RangedLayoutReader>> {
        self.children()
            .zip_eq(self.child_ranges())
            .map(|((i, c), (begin, end))| {
                let layout = self.layout_builder.read_layout(
                    self.fb_bytes.clone(),
                    c._tab.loc(),
                    self.scan.clone(),
                    cache(i as u16),
                )?;
                Ok(((begin, end), layout))
            })
            .collect::<VortexResult<VecDeque<_>>>()
    }
}

impl LayoutReader for ChunkedLayout {
    fn add_splits(&self, row_offset: usize, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
        for ((begin, _), child) in self.child_layouts(|i| self.message_cache.unknown_dtype(i))? {
            child.add_splits(row_offset + begin, splits)?
        }
        Ok(())
    }

    fn read_selection(&mut self, selector: &RowMask) -> VortexResult<Option<BatchRead>> {
        if let Some(br) = &mut self.chunk_reader {
            br.read_next(selector)
        } else {
            self.chunk_reader = Some(BufferedLayoutReader::new(self.child_layouts(|i| {
                self.message_cache
                    .relative(i, self.message_cache.dtype().clone())
            })?));
            self.read_selection(selector)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::iter;
    use std::sync::{Arc, RwLock};

    use bytes::Bytes;
    use croaring::Bitmap;
    use flatbuffers::{root_unchecked, FlatBufferBuilder};
    use futures_util::TryStreamExt;
    use vortex_array::array::{ChunkedArray, PrimitiveArray};
    use vortex_array::{ArrayDType, IntoArrayData, IntoArrayVariant};
    use vortex_dtype::PType;
    use vortex_expr::{BinaryExpr, Identity, Literal, Operator};
    use vortex_flatbuffers::{footer, WriteFlatBuffer};
    use vortex_ipc::messages::writer::MessageWriter;
    use vortex_ipc::stream_writer::ByteRange;

    use crate::read::cache::{LazilyDeserializedDType, RelativeLayoutCache};
    use crate::read::layouts::chunked::ChunkedLayout;
    use crate::read::layouts::test_read::{filter_read_layout, read_layout, read_layout_data};
    use crate::read::mask::RowMask;
    use crate::{write, LayoutDeserializer, LayoutMessageCache, RowFilter, Scan};

    async fn layout_and_bytes(
        cache: Arc<RwLock<LayoutMessageCache>>,
        scan: Scan,
    ) -> (ChunkedLayout, ChunkedLayout, Bytes, usize) {
        let mut writer = MessageWriter::new(Vec::new());
        let array = PrimitiveArray::from((0..100).collect::<Vec<_>>()).into_array();
        let array_dtype = array.dtype().clone();
        let chunked =
            ChunkedArray::try_new(iter::repeat(array).take(5).collect(), array_dtype).unwrap();
        let len = chunked.len();
        let mut byte_offsets = vec![writer.tell()];
        let mut row_offsets = vec![0];
        let mut row_offset = 0;

        let mut chunk_stream = chunked.array_stream();
        while let Some(chunk) = chunk_stream.try_next().await.unwrap() {
            row_offset += chunk.len() as u64;
            row_offsets.push(row_offset);
            writer.write_batch(chunk).await.unwrap();
            byte_offsets.push(writer.tell());
        }
        let flat_layouts = byte_offsets
            .iter()
            .zip(byte_offsets.iter().skip(1))
            .zip(
                row_offsets
                    .iter()
                    .zip(row_offsets.iter().skip(1))
                    .map(|(begin, end)| end - begin),
            )
            .map(|((begin, end), len)| write::Layout::flat(ByteRange::new(*begin, *end), len))
            .collect::<VecDeque<_>>();

        row_offsets.truncate(row_offsets.len() - 1);

        let written = writer.into_inner();

        let mut fb = FlatBufferBuilder::new();
        let chunked_layout = write::Layout::chunked(flat_layouts.into(), len as u64, false);
        let flat_buf = chunked_layout.write_flatbuffer(&mut fb);
        fb.finish_minimal(flat_buf);
        let fb_bytes = Bytes::copy_from_slice(fb.finished_data());

        let fb_loc = (unsafe { root_unchecked::<footer::Layout>(&fb_bytes) })
            ._tab
            .loc();

        let dtype = Arc::new(LazilyDeserializedDType::from_dtype(PType::I32.into()));
        (
            ChunkedLayout::new(
                fb_bytes.clone(),
                fb_loc,
                scan,
                LayoutDeserializer::default(),
                RelativeLayoutCache::new(cache.clone(), dtype.clone()),
            ),
            ChunkedLayout::new(
                fb_bytes,
                fb_loc,
                Scan::new(None),
                LayoutDeserializer::default(),
                RelativeLayoutCache::new(cache, dtype),
            ),
            Bytes::from(written),
            len,
        )
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_range() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (mut filter_layout, mut projection_layout, buf, length) = layout_and_bytes(
            cache.clone(),
            Scan::new(Some(RowFilter::new_expr(BinaryExpr::new_expr(
                Arc::new(Identity),
                Operator::Gt,
                Literal::new_expr(10.into()),
            )))),
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
        let (_, mut projection_layout, buf, length) =
            layout_and_bytes(cache.clone(), Scan::new(None)).await;
        let arr = read_layout(&mut projection_layout, cache, &buf, length).pop_front();

        assert!(arr.is_some());
        let arr = arr.unwrap();
        assert_eq!(
            arr.into_primitive().unwrap().maybe_null_slice::<i32>(),
            (0..100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_no_range() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (_, mut projection_layout, buf, _) =
            layout_and_bytes(cache.clone(), Scan::new(None)).await;
        let arr = read_layout_data(
            &mut projection_layout,
            cache,
            &buf,
            &RowMask::try_new(Bitmap::from_range(0..500), 0, 500).unwrap(),
        );

        assert!(arr.is_some());
        let arr = arr.unwrap();
        assert_eq!(
            arr.into_primitive().unwrap().maybe_null_slice::<i32>(),
            iter::repeat(0..100).take(5).flatten().collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_multiple_selectors() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (_, mut projection_layout, buf, _) =
            layout_and_bytes(cache.clone(), Scan::new(None)).await;
        let mut arr = [
            RowMask::try_new(Bitmap::from_range(0..150), 0, 200).unwrap(),
            RowMask::try_new(Bitmap::from_range(50..150), 200, 400).unwrap(),
            RowMask::try_new(Bitmap::from_range(0..100), 400, 500).unwrap(),
        ]
        .into_iter()
        .flat_map(|s| read_layout_data(&mut projection_layout, cache.clone(), &buf, &s))
        .collect::<VecDeque<_>>();

        assert_eq!(arr.len(), 3);
        assert_eq!(
            arr.pop_front()
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(0..100).chain(0..50).collect::<Vec<_>>()
        );
        assert_eq!(
            arr.pop_front()
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(50..100).chain(0..50).collect::<Vec<_>>()
        );
        assert_eq!(
            arr.pop_front()
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(0..100).collect::<Vec<_>>()
        );
    }
}
