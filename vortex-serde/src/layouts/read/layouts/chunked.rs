use std::collections::VecDeque;

use bytes::Bytes;
use croaring::Bitmap;
use itertools::Itertools;
use vortex::{Array, IntoArrayVariant};
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_flatbuffers::footer;

use crate::layouts::read::buffered::{BufferedLayoutReader, RangedLayoutReader};
use crate::layouts::read::cache::RelativeLayoutCache;
use crate::layouts::read::selection::RowSelector;
use crate::layouts::{
    LayoutDeserializer, LayoutId, LayoutReader, LayoutSpec, Message, RangeResult, ReadResult, Scan,
    CHUNKED_LAYOUT_ID,
};
#[derive(Default, Debug)]
pub struct ChunkedLayoutSpec;

impl LayoutSpec for ChunkedLayoutSpec {
    fn id(&self) -> LayoutId {
        CHUNKED_LAYOUT_ID
    }

    fn layout(
        &self,
        fb_bytes: Bytes,
        fb_loc: usize,
        length: u64,
        scan: Scan,
        layout_builder: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> Box<dyn LayoutReader> {
        Box::new(ChunkedLayout::new(
            fb_bytes,
            fb_loc,
            length,
            scan,
            layout_builder,
            message_cache,
        ))
    }
}

#[derive(Debug)]
pub enum MetadataState {
    Init,
    ReadMetadata((Box<dyn LayoutReader>, usize)),
    Array(Array),
}

/// In memory representation of Chunked NestedLayout.
///
/// First child in the list is the metadata table
/// Subsequent children are consecutive chunks of this layout
#[derive(Debug)]
pub struct ChunkedLayout {
    fb_bytes: Bytes,
    fb_loc: usize,
    length: u64,
    offset: usize,
    scan: Scan,
    layout_builder: LayoutDeserializer,
    message_cache: RelativeLayoutCache,
    chunk_reader: Option<BufferedLayoutReader>,
    metadata_array: MetadataState,
}

impl ChunkedLayout {
    pub fn new(
        fb_bytes: Bytes,
        fb_loc: usize,
        length: u64,
        scan: Scan,
        layout_builder: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> Self {
        Self {
            fb_bytes,
            fb_loc,
            length,
            offset: 0,
            scan,
            layout_builder,
            message_cache,
            chunk_reader: None,
            metadata_array: MetadataState::Init,
        }
    }

    fn flatbuffer(&self) -> footer::Layout {
        unsafe {
            let tab = flatbuffers::Table::new(&self.fb_bytes, self.fb_loc);
            footer::Layout::init_from_table(tab)
        }
    }

    fn child_ranges(&self) -> VortexResult<Vec<(usize, usize)>> {
        let MetadataState::Array(ref m) = self.metadata_array else {
            vortex_bail!("Must fetch metadata before")
        };

        let row_offset = m
            .with_dyn(|a| a.as_struct_array_unchecked().field(0))
            .ok_or_else(|| vortex_err!("must have row_offset metadata column"))?;
        let primitive_offsets = row_offset.into_primitive()?;
        Ok(primitive_offsets
            .maybe_null_slice::<u64>()
            .iter()
            .chain(&[self.length])
            .tuple_windows()
            .map(|(begin, end)| (*begin as usize, *end as usize))
            .collect::<Vec<_>>())
    }

    fn ranged_children(&self) -> VortexResult<VecDeque<RangedLayoutReader>> {
        let dtype = self.message_cache.dtype();
        self.flatbuffer()
            .children()
            .ok_or_else(|| vortex_err!("Missing children"))?
            .iter()
            .enumerate()
            // Skip over the metadata table of this layout
            .skip(if self.has_metadata() { 1 } else { 0 })
            .zip_eq(self.child_ranges()?)
            .skip_while(|(_, (_, end))| *end < self.offset)
            .map(|((i, c), (begin, end))| {
                let mut layout = self.layout_builder.read_layout(
                    self.fb_bytes.clone(),
                    c._tab.loc(),
                    (end - begin) as u64,
                    self.scan.clone(),
                    self.message_cache.relative(i as u16, dtype.clone()),
                )?;
                if self.offset > begin {
                    layout.advance(self.offset - begin)?;
                }
                Ok(((begin, end), layout))
            })
            .collect::<VortexResult<VecDeque<_>>>()
    }

    fn metadata_layout(&self) -> VortexResult<(Box<dyn LayoutReader>, usize)> {
        let children = self
            .flatbuffer()
            .children()
            .ok_or_else(|| vortex_err!("Missing children"))?;
        let metadata_child = children.get(0);
        Ok((
            self.layout_builder.read_layout(
                self.fb_bytes.clone(),
                metadata_child._tab.loc(),
                children.len() as u64,
                Scan::new(None),
                self.message_cache.stored_dtype(0u16),
            )?,
            children.len() - 1,
        ))
    }

    fn has_metadata(&self) -> bool {
        self.flatbuffer()
            .metadata()
            .map(|b| b.bytes()[0] != 0)
            .unwrap_or(false)
    }
}

impl LayoutReader for ChunkedLayout {
    fn next_range(&mut self) -> VortexResult<RangeResult> {
        if let Some(br) = &mut self.chunk_reader {
            br.next_range()
        } else {
            match &mut self.metadata_array {
                MetadataState::Init => {
                    self.metadata_array = MetadataState::ReadMetadata(self.metadata_layout()?);
                    self.next_range()
                }
                MetadataState::ReadMetadata((mr, nchildren)) => {
                    match mr.read_next(RowSelector::new(
                        Bitmap::from_range(0..*nchildren as u32),
                        0,
                        *nchildren,
                    ))? {
                        None => {
                            unreachable!(
                                "Metadata isn't chunked, will terminate after first batch result"
                            )
                        }
                        Some(mr) => match mr {
                            ReadResult::ReadMore(m) => Ok(RangeResult::ReadMore(m)),
                            ReadResult::Batch(r) => {
                                if matches!(self.metadata_array, MetadataState::Array(_)) {
                                    vortex_bail!("Metadata is not chunked for now");
                                } else {
                                    self.metadata_array = MetadataState::Array(r);
                                    self.chunk_reader =
                                        Some(BufferedLayoutReader::new(self.ranged_children()?));
                                }
                                self.next_range()
                            }
                        },
                    }
                }
                MetadataState::Array(_) => {
                    unreachable!("Already fetched metadata but didn't create reader")
                }
            }
        }
    }

    fn read_next(&mut self, selector: RowSelector) -> VortexResult<Option<ReadResult>> {
        if let Some(br) = &mut self.chunk_reader {
            br.read_next(selector)
        } else {
            match &mut self.metadata_array {
                MetadataState::Init => {
                    self.metadata_array = MetadataState::ReadMetadata(self.metadata_layout()?);
                    self.read_next(selector)
                }
                MetadataState::ReadMetadata((mr, nchildren)) => {
                    match mr.read_next(RowSelector::new(
                        Bitmap::from_range(0..*nchildren as u32),
                        0,
                        *nchildren,
                    ))? {
                        None => {
                            unreachable!(
                                "Metadata isn't chunked, will terminate after first batch result"
                            )
                        }
                        Some(mr) => match mr {
                            ReadResult::ReadMore(m) => Ok(Some(ReadResult::ReadMore(m))),
                            ReadResult::Batch(r) => {
                                if matches!(self.metadata_array, MetadataState::Array(_)) {
                                    vortex_bail!("Metadata is not chunked for now");
                                } else {
                                    self.metadata_array = MetadataState::Array(r);
                                    self.chunk_reader =
                                        Some(BufferedLayoutReader::new(self.ranged_children()?));
                                }
                                self.read_next(selector)
                            }
                        },
                    }
                }
                MetadataState::Array(_) => {
                    unreachable!("Already fetched metadata but didn't create reader")
                }
            }
        }
    }

    fn advance(&mut self, up_to_row: usize) -> VortexResult<Vec<Message>> {
        if let Some(br) = &mut self.chunk_reader {
            br.advance(up_to_row)
        } else {
            self.offset = up_to_row;
            Ok(Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::iter;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, RwLock};

    use bytes::Bytes;
    use croaring::Bitmap;
    use flatbuffers::{root_unchecked, FlatBufferBuilder};
    use futures_util::TryStreamExt;
    use vortex::array::{ChunkedArray, PrimitiveArray, StructArray};
    use vortex::validity::Validity;
    use vortex::{ArrayDType, IntoArray, IntoArrayVariant};
    use vortex_dtype::PType;
    use vortex_expr::{BinaryExpr, Identity, Literal, Operator};
    use vortex_flatbuffers::{footer, WriteFlatBuffer};

    use crate::layouts::read::cache::{LazyDeserializedDType, RelativeLayoutCache};
    use crate::layouts::read::layouts::chunked::ChunkedLayout;
    use crate::layouts::read::layouts::test_read::{
        filter_read_layout, read_filters, read_layout, read_layout_data, read_layout_ranges,
    };
    use crate::layouts::read::selection::RowSelector;
    use crate::layouts::{
        write, LayoutDeserializer, LayoutMessageCache, LayoutReader, RowFilter, Scan,
    };
    use crate::message_writer::MessageWriter;
    use crate::stream_writer::ByteRange;

    async fn layout_and_bytes(
        cache: Arc<RwLock<LayoutMessageCache>>,
        scan: Scan,
    ) -> (ChunkedLayout, ChunkedLayout, Bytes) {
        let mut writer = MessageWriter::new(Vec::new());
        let array = PrimitiveArray::from((0..100).collect::<Vec<_>>()).into_array();
        let array_dtype = array.dtype().clone();
        let chunked =
            ChunkedArray::try_new(iter::repeat(array).take(5).collect(), array_dtype).unwrap();
        let len = chunked.len() as u64;
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
        let mut flat_layouts = byte_offsets
            .iter()
            .zip(byte_offsets.iter().skip(1))
            .map(|(begin, end)| write::Layout::flat(ByteRange::new(*begin, *end)))
            .collect::<VecDeque<_>>();

        row_offsets.truncate(row_offsets.len() - 1);

        let meta_len = row_offsets.len();
        let metadata_array = StructArray::try_new(
            ["row_offset".into()].into(),
            vec![row_offsets.into_array()],
            meta_len,
            Validity::NonNullable,
        )
        .unwrap();

        let dtype_begin = writer.tell();
        writer.write_dtype(metadata_array.dtype()).await.unwrap();
        let dtype_end = writer.tell();
        writer
            .write_batch(metadata_array.into_array())
            .await
            .unwrap();
        flat_layouts.push_front(write::Layout::inlined_schema(
            vec![write::Layout::flat(ByteRange::new(
                dtype_end,
                writer.tell(),
            ))],
            ByteRange::new(dtype_begin, dtype_end),
        ));

        let written = writer.into_inner();

        let mut fb = FlatBufferBuilder::new();
        let chunked_layout = write::Layout::chunked(flat_layouts.into(), true);
        let flat_buf = chunked_layout.write_flatbuffer(&mut fb);
        fb.finish_minimal(flat_buf);
        let fb_bytes = Bytes::copy_from_slice(fb.finished_data());

        let fb_loc = (unsafe { root_unchecked::<footer::Layout>(&fb_bytes) })
            ._tab
            .loc();

        let dtype = Arc::new(LazyDeserializedDType::from_dtype(PType::I32.into()));
        (
            ChunkedLayout::new(
                fb_bytes.clone(),
                fb_loc,
                len,
                scan,
                LayoutDeserializer::default(),
                RelativeLayoutCache::new(cache.clone(), dtype.clone()),
            ),
            ChunkedLayout::new(
                fb_bytes,
                fb_loc,
                len,
                Scan::new(None),
                LayoutDeserializer::default(),
                RelativeLayoutCache::new(cache, dtype),
            ),
            Bytes::from(written),
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
        let (_, mut projection_layout, buf) =
            layout_and_bytes(cache.clone(), Scan::new(None)).await;
        let arr = read_layout(&mut projection_layout, cache, &buf).pop_front();

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
        let (_, mut projection_layout, buf) =
            layout_and_bytes(cache.clone(), Scan::new(None)).await;
        let arr = read_layout_data(
            &mut projection_layout,
            cache,
            &buf,
            RowSelector::new(Bitmap::from_range(0..500), 0, 500),
        )
        .pop();

        assert!(arr.is_some());
        let arr = arr.unwrap();
        assert_eq!(
            arr.into_primitive().unwrap().maybe_null_slice::<i32>(),
            iter::repeat(0..100).take(5).flatten().collect::<Vec<_>>()
        );
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
        filter_layout.advance(500).unwrap();
        let arr = filter_read_layout(&mut filter_layout, &mut projection_layout, cache, &buf);

        assert!(arr.is_empty());
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn read_multiple_selectors() {
        let cache = Arc::new(RwLock::new(LayoutMessageCache::default()));
        let (_, mut projection_layout, buf) =
            layout_and_bytes(cache.clone(), Scan::new(None)).await;
        let mut arr = [
            RowSelector::new(Bitmap::from_range(0..150), 0, 200),
            RowSelector::new(Bitmap::from_range(250..350), 200, 400),
            RowSelector::new(Bitmap::from_range(400..500), 400, 500),
        ]
        .into_iter()
        .flat_map(|s| read_layout_data(&mut projection_layout, cache.clone(), &buf, s))
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
        let mut arr = selector
            .into_iter()
            .flat_map(|s| read_layout_data(&mut projection_layout, cache.clone(), &buf, s))
            .collect::<VecDeque<_>>();

        assert_eq!(arr.len(), 5);
        assert_eq!(
            arr.pop_front()
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(50..100).collect::<Vec<_>>()
        );
        assert_eq!(
            arr[3]
                .clone()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(11..100).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn advance_mid_read() {
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
        let selectors = read_layout_ranges(&mut filter_layout, cache.clone(), &buf)
            .into_iter()
            .flat_map(|s| read_filters(&mut filter_layout, cache.clone(), &buf, s))
            .collect::<Vec<_>>();
        let advanced = AtomicBool::new(false);
        let mut arr = selectors
            .into_iter()
            .flat_map(|s| {
                let a = read_layout_data(&mut projection_layout, cache.clone(), &buf, s);
                if advanced
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    projection_layout.advance(321).unwrap();
                }
                a
            })
            .collect::<VecDeque<_>>();

        assert_eq!(arr.len(), 3);
        assert_eq!(
            arr.pop_front()
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(11..100).collect::<Vec<_>>()
        );
        assert_eq!(
            arr.pop_front()
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(21..100).collect::<Vec<_>>()
        );
        assert_eq!(
            arr.pop_front()
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            &(11..100).collect::<Vec<_>>()
        );
    }
}
