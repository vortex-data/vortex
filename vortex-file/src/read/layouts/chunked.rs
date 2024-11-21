use std::collections::BTreeSet;
use std::sync::RwLock;

use bytes::Bytes;
use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::array::ChunkedArray;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::{vortex_err, vortex_panic, VortexResult};
use vortex_flatbuffers::footer;

use crate::layouts::RangedLayoutReader;
use crate::read::cache::RelativeLayoutCache;
use crate::read::mask::RowMask;
use crate::{
    BatchRead, Layout, LayoutDeserializer, LayoutId, LayoutPartId, LayoutReader, MessageLocator,
    Scan, CHUNKED_LAYOUT_ID,
};

#[derive(Default, Debug)]
pub struct ChunkedLayout;

/// In memory representation of Chunked NestedLayout.
///
/// First child in the list is the metadata table
/// Subsequent children are consecutive chunks of this layout
impl Layout for ChunkedLayout {
    fn id(&self) -> LayoutId {
        CHUNKED_LAYOUT_ID
    }

    fn reader(
        &self,
        fb_bytes: Bytes,
        fb_loc: usize,
        scan: Scan,
        layout_builder: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> VortexResult<Box<dyn LayoutReader>> {
        Ok(Box::new(
            ChunkedLayoutBuilder {
                fb_bytes,
                fb_loc,
                scan,
                layout_builder,
                message_cache,
            }
            .build()?,
        ))
    }
}

const METADATA_LAYOUT_PART_ID: LayoutPartId = 0;

/// In memory representation of Chunked NestedLayout.
///
/// First child in the list is the metadata table
/// Subsequent children are consecutive chunks of this layout
struct ChunkedLayoutBuilder {
    fb_bytes: Bytes,
    fb_loc: usize,
    scan: Scan,
    layout_builder: LayoutDeserializer,
    message_cache: RelativeLayoutCache,
}

impl ChunkedLayoutBuilder {
    fn flatbuffer(&self) -> footer::Layout {
        unsafe {
            let tab = flatbuffers::Table::new(&self.fb_bytes, self.fb_loc);
            footer::Layout::init_from_table(tab)
        }
    }

    fn metadata_layout(&self) -> VortexResult<Option<Box<dyn LayoutReader>>> {
        self.has_metadata()
            .then(|| {
                let metadata_fb = self
                    .flatbuffer()
                    .children()
                    .ok_or_else(|| vortex_err!("must have metadata"))?
                    .get(0);
                self.layout_builder.read_layout(
                    self.fb_bytes.clone(),
                    metadata_fb._tab.loc(),
                    // TODO(robert): Create stats projection
                    Scan::new(None),
                    self.message_cache.unknown_dtype(METADATA_LAYOUT_PART_ID),
                )
            })
            .transpose()
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

    fn children_ranges(&self) -> Vec<(usize, usize)> {
        self.children()
            .map(|(_, c)| c.row_count())
            .scan(0u64, |acc, row_count| {
                let current = *acc;
                *acc += row_count;
                Some((current as usize, *acc as usize))
            })
            .collect::<Vec<_>>()
    }

    fn children_layouts(&self) -> VortexResult<Vec<RangedLayoutReader>> {
        self.children()
            .zip_eq(self.children_ranges())
            .map(|((i, c), (begin, end))| {
                let layout = self.layout_builder.read_layout(
                    self.fb_bytes.clone(),
                    c._tab.loc(),
                    self.scan.clone(),
                    self.message_cache
                        .relative(i as u16, self.message_cache.dtype().clone()),
                )?;
                Ok(((begin, end), layout))
            })
            .collect::<VortexResult<Vec<_>>>()
    }

    pub fn build(&self) -> VortexResult<ChunkedLayoutReader> {
        Ok(ChunkedLayoutReader::new(
            self.children_layouts()?,
            self.metadata_layout()?,
        ))
    }
}

#[derive(Debug, Default, Clone)]
enum ChildRead {
    #[default]
    NotStarted,
    Finished(Option<ArrayData>),
}

impl ChildRead {
    pub fn finished(&self) -> bool {
        matches!(self, Self::Finished(_))
    }

    pub fn into_value(self) -> Option<ArrayData> {
        match self {
            ChildRead::NotStarted => None,
            ChildRead::Finished(v) => v,
        }
    }
}

type InProgressLayoutRanges = RwLock<HashMap<(usize, usize), (Vec<usize>, Vec<ChildRead>)>>;

#[allow(dead_code)]
#[derive(Debug)]
pub struct ChunkedLayoutReader {
    layouts: Vec<RangedLayoutReader>,
    metadata_layout: Option<Box<dyn LayoutReader>>,
    in_progress_ranges: InProgressLayoutRanges,
}

impl ChunkedLayoutReader {
    pub fn new(
        layouts: Vec<RangedLayoutReader>,
        metadata_layout: Option<Box<dyn LayoutReader>>,
    ) -> Self {
        Self {
            layouts,
            metadata_layout,
            in_progress_ranges: RwLock::new(HashMap::new()),
        }
    }

    fn buffer_read(&self, mask: &RowMask) -> VortexResult<Vec<MessageLocator>> {
        let mut in_progress_guard = self
            .in_progress_ranges
            .write()
            .unwrap_or_else(|poison| vortex_panic!("Failed to write to message cache: {poison}"));
        let (layout_idxs, in_progress_range) = in_progress_guard
            .entry((mask.begin(), mask.end()))
            .or_insert_with(|| {
                let layouts_in_range = self
                    .layouts
                    .iter()
                    .enumerate()
                    .filter_map(|(i, ((begin, end), _))| {
                        (mask.end() > *begin && mask.begin() < *end).then_some(i)
                    })
                    .collect::<Vec<_>>();
                let num_layouts = layouts_in_range.len();
                (layouts_in_range, vec![ChildRead::default(); num_layouts])
            });

        let mut messages_to_fetch = Vec::new();
        for (((begin, end), layout), array_slot) in layout_idxs
            .iter()
            .map(|i| &self.layouts[*i])
            .zip(in_progress_range)
            .filter(|(_, cr)| !cr.finished())
        {
            let layout_selection = mask.slice(*begin, *end).shift(*begin)?;
            if let Some(rr) = layout.read_selection(&layout_selection)? {
                match rr {
                    BatchRead::ReadMore(m) => {
                        messages_to_fetch.extend(m);
                    }
                    BatchRead::Batch(a) => {
                        *array_slot = ChildRead::Finished(Some(a));
                    }
                }
            } else {
                *array_slot = ChildRead::Finished(None);
            }
        }

        Ok(messages_to_fetch)
    }

    #[allow(dead_code)]
    pub fn n_chunks(&self) -> usize {
        self.layouts.len()
    }

    #[allow(dead_code)]
    pub fn metadata_layout(&self) -> Option<&dyn LayoutReader> {
        self.metadata_layout.as_deref()
    }
}

impl LayoutReader for ChunkedLayoutReader {
    fn add_splits(&self, row_offset: usize, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
        for ((begin, _), child) in &self.layouts {
            child.add_splits(row_offset + begin, splits)?
        }
        Ok(())
    }

    fn read_selection(&self, selector: &RowMask) -> VortexResult<Option<BatchRead>> {
        let messages_to_fetch = self.buffer_read(selector)?;
        if !messages_to_fetch.is_empty() {
            return Ok(Some(BatchRead::ReadMore(messages_to_fetch)));
        }

        if let Some((_, arrays_in_range)) = self
            .in_progress_ranges
            .write()
            .unwrap_or_else(|poison| vortex_panic!("Failed to write to message cache: {poison}"))
            .remove(&(selector.begin(), selector.end()))
        {
            let mut child_arrays = arrays_in_range
                .into_iter()
                .filter_map(ChildRead::into_value)
                .collect::<Vec<_>>();
            match child_arrays.len() {
                0 | 1 => Ok(child_arrays.pop().map(BatchRead::Batch)),
                _ => {
                    let dtype = child_arrays[0].dtype().clone();
                    Ok(Some(BatchRead::Batch(
                        ChunkedArray::try_new(child_arrays, dtype)?.into_array(),
                    )))
                }
            }
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::iter;
    use std::sync::{Arc, RwLock};

    use arrow_buffer::BooleanBufferBuilder;
    use bytes::Bytes;
    use flatbuffers::{root_unchecked, FlatBufferBuilder};
    use futures_util::TryStreamExt;
    use vortex_array::array::{BoolArray, ChunkedArray, PrimitiveArray};
    use vortex_array::{ArrayDType, ArrayLen, IntoArrayData, IntoArrayVariant};
    use vortex_dtype::{Nullability, PType};
    use vortex_expr::{BinaryExpr, Identity, Literal, Operator};
    use vortex_flatbuffers::{footer, WriteFlatBuffer};
    use vortex_ipc::messages::writer::MessageWriter;
    use vortex_ipc::stream_writer::ByteRange;

    use crate::layouts::chunked::{ChunkedLayoutBuilder, ChunkedLayoutReader};
    use crate::read::cache::{LazyDType, RelativeLayoutCache};
    use crate::read::layouts::test_read::{filter_read_layout, read_layout, read_layout_data};
    use crate::read::mask::RowMask;
    use crate::{write, LayoutDeserializer, LayoutMessageCache, RowFilter, Scan};

    async fn layout_and_bytes(
        cache: Arc<RwLock<LayoutMessageCache>>,
        scan: Scan,
    ) -> (ChunkedLayoutReader, ChunkedLayoutReader, Bytes, usize) {
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
            .map(|((begin, end), len)| write::LayoutSpec::flat(ByteRange::new(*begin, *end), len))
            .collect::<VecDeque<_>>();

        row_offsets.truncate(row_offsets.len() - 1);

        let written = writer.into_inner();

        let mut fb = FlatBufferBuilder::new();
        let chunked_layout = write::LayoutSpec::chunked(flat_layouts.into(), len as u64, false);
        let flat_buf = chunked_layout.write_flatbuffer(&mut fb);
        fb.finish_minimal(flat_buf);
        let fb_bytes = Bytes::copy_from_slice(fb.finished_data());

        let fb_loc = (unsafe { root_unchecked::<footer::Layout>(&fb_bytes) })
            ._tab
            .loc();

        let dtype = Arc::new(LazyDType::from_dtype(PType::I32.into()));
        let layout_builder = LayoutDeserializer::default();
        (
            ChunkedLayoutBuilder {
                fb_bytes: fb_bytes.clone(),
                fb_loc,
                scan,
                layout_builder: layout_builder.clone(),
                message_cache: RelativeLayoutCache::new(cache.clone(), dtype.clone()),
            }
            .build()
            .unwrap(),
            ChunkedLayoutBuilder {
                fb_bytes,
                fb_loc,
                scan: Scan::new(None),
                layout_builder,
                message_cache: RelativeLayoutCache::new(cache, dtype),
            }
            .build()
            .unwrap(),
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

        assert_eq!(filter_layout.n_chunks(), 5);
        assert_eq!(projection_layout.n_chunks(), 5);

        assert!(filter_layout.metadata_layout().is_none());
        assert!(projection_layout.metadata_layout().is_none());

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
            &RowMask::new_valid_between(0, 500),
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

        let mut first_range = BooleanBufferBuilder::new(200);
        first_range.append_n(150, true);
        first_range.append_n(50, false);

        let mut snd_range = BooleanBufferBuilder::new(200);
        snd_range.append_n(50, false);
        snd_range.append_n(100, true);
        snd_range.append_n(50, false);
        let mut arr = [
            RowMask::try_new(
                BoolArray::new(first_range.finish(), Nullability::NonNullable).into_array(),
                0,
                200,
            )
            .unwrap(),
            RowMask::try_new(
                BoolArray::new(snd_range.finish(), Nullability::NonNullable).into_array(),
                200,
                400,
            )
            .unwrap(),
            RowMask::new_valid_between(400, 500),
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
