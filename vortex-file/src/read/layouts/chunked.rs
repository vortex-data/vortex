use std::collections::BTreeSet;
use std::sync::{Arc, OnceLock, RwLock};

use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::array::ChunkedArray;
use vortex_array::compute::{scalar_at, take};
use vortex_array::stats::{stats_from_bitset_bytes, ArrayStatistics as _, Stat};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_dtype::field::Field;
use vortex_dtype::{DType, Nullability, StructDType};
use vortex_error::{
    vortex_bail, vortex_err, vortex_panic, VortexExpect as _, VortexResult, VortexUnwrap,
};
use vortex_expr::Select;
use vortex_flatbuffers::footer as fb;

use crate::layouts::RangedLayoutReader;
use crate::pruning::PruningPredicate;
use crate::read::cache::RelativeLayoutCache;
use crate::read::mask::RowMask;
use crate::{
    Layout, LayoutDeserializer, LayoutId, LayoutPartId, LayoutReader, LazyDType, MessageLocator,
    PollRead, Prune, Scan, CHUNKED_LAYOUT_ID,
};

#[derive(Default, Debug)]
pub struct ChunkedLayout;

/// In-memory representation of Chunked layout.
///
/// First child in the list is the metadata table
/// Subsequent children are consecutive chunks of this layout
impl Layout for ChunkedLayout {
    fn id(&self) -> LayoutId {
        CHUNKED_LAYOUT_ID
    }

    fn reader(
        &self,
        layout: fb::Layout,
        scan: Scan,
        layout_builder: LayoutDeserializer,
        message_cache: RelativeLayoutCache,
    ) -> VortexResult<Box<dyn LayoutReader>> {
        Ok(Box::new(
            ChunkedLayoutBuilder {
                layout,
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
struct ChunkedLayoutBuilder<'a> {
    layout: fb::Layout<'a>,
    scan: Scan,
    layout_builder: LayoutDeserializer,
    message_cache: RelativeLayoutCache,
}

impl ChunkedLayoutBuilder<'_> {
    pub fn build(&self) -> VortexResult<ChunkedLayoutReader> {
        // If the metadata bytes of the layout are present, interpret them as a bitset of `Stat`s,
        // and read the first child layout as a table with each stat as a column and each row
        // as the stat value for the N-th chunk.
        let stats_layout = if let Some(metadata) = self.layout.metadata() {
            let set_stats = stats_from_bitset_bytes(metadata.bytes());
            let metadata_fb = self
                .layout
                .children()
                .ok_or_else(|| vortex_err!("Must have children if layout has metadata"))?
                .get(0);
            let stats_dtype = stats_table_dtype(&set_stats, self.message_cache.dtype().value()?);
            let DType::Struct(ref s, _) = stats_dtype else {
                vortex_bail!("Chunked layout stats must be a Struct, got {stats_dtype}")
            };
            Some(self.layout_builder.read_layout(
                metadata_fb,
                Scan::new(Some(Arc::new(Select::include(
                    s.names().iter().map(|s| Field::Name(s.clone())).collect(),
                )))),
                self.message_cache.relative(
                    METADATA_LAYOUT_PART_ID,
                    Arc::new(LazyDType::from_dtype(stats_dtype)),
                ),
            )?)
        } else {
            None
        };

        // Prepare the layouts for each of the children (chunks).
        // This will start at the 0th child if there are no chunk stats, and the 1st child otherwise.
        let chunk_layouts: Vec<RangedLayoutReader> = self
            .layout
            .children()
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .skip(if stats_layout.is_some() { 1 } else { 0 })
            .scan(0usize, |total_rows, (child_idx, next_chunk)| {
                // Calculate the start/end range of the chunk in the global row offset range.
                let chunk_start = *total_rows;
                *total_rows += usize::try_from(next_chunk.row_count()).vortex_expect("row_count");
                let chunk_end = *total_rows;

                // Relative layout cache for the `child_idx`-th child.
                let child_cache = self.message_cache.relative(
                    child_idx.try_into().vortex_unwrap(),
                    self.message_cache.dtype().clone(),
                );

                // Construct the ranged layout.
                Some(
                    self.layout_builder
                        .read_layout(next_chunk, self.scan.clone(), child_cache)
                        .map(|layout| RangedLayoutReader((chunk_start, chunk_end), layout)),
                )
            })
            .try_collect()?;

        Ok(ChunkedLayoutReader::new(
            chunk_layouts,
            stats_layout,
            self.scan.clone(),
        ))
    }
}

fn stats_table_dtype(stats: &[Stat], dtype: &DType) -> DType {
    let dtypes = stats.iter().map(|s| s.dtype(dtype).as_nullable()).collect();

    DType::Struct(
        StructDType::new(stats.iter().map(|s| s.name().into()).collect(), dtypes),
        Nullability::NonNullable,
    )
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
    scan: Scan,
    in_progress_ranges: InProgressLayoutRanges,
    cached_metadata: OnceLock<ArrayData>,
    cached_prunability: OnceLock<ArrayData>,
}

impl ChunkedLayoutReader {
    pub fn new(
        layouts: Vec<RangedLayoutReader>,
        metadata_layout: Option<Box<dyn LayoutReader>>,
        scan: Scan,
    ) -> Self {
        Self {
            layouts,
            metadata_layout,
            scan,
            in_progress_ranges: RwLock::new(HashMap::new()),
            cached_metadata: OnceLock::new(),
            cached_prunability: OnceLock::new(),
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
                let layouts_in_range = self.children_for_row_range(mask.begin(), mask.end());
                let num_layouts = layouts_in_range.len();
                (layouts_in_range, vec![ChildRead::default(); num_layouts])
            });

        let mut messages_to_fetch = Vec::new();
        for (RangedLayoutReader((begin, end), layout), array_slot) in layout_idxs
            .iter()
            .map(|i| &self.layouts[*i])
            .zip(in_progress_range)
            .filter(|(_, cr)| !cr.finished())
        {
            let layout_selection = mask.slice(*begin, *end)?.shift(*begin)?;
            if let Some(rr) = layout.poll_read(&layout_selection)? {
                match rr {
                    PollRead::ReadMore(m) => {
                        messages_to_fetch.extend(m);
                    }
                    PollRead::Value(a) => {
                        *array_slot = ChildRead::Finished(Some(a));
                    }
                }
            } else {
                *array_slot = ChildRead::Finished(None);
            }
        }

        Ok(messages_to_fetch)
    }

    pub fn n_chunks(&self) -> usize {
        self.layouts.len()
    }

    pub fn metadata_layout(&self) -> Option<&dyn LayoutReader> {
        self.metadata_layout.as_deref()
    }

    /// Return the index for all chunks which contain rows begin
    /// `begin` (inclusive) and `end` (exclusive).
    fn children_for_row_range(&self, begin: usize, end: usize) -> Vec<usize> {
        self.layouts
            .iter()
            .enumerate()
            .filter_map(|(i, RangedLayoutReader((child_begin, child_end), _))| {
                (end > *child_begin && begin < *child_end).then_some(i)
            })
            .collect::<Vec<_>>()
    }

    fn can_prune_overlapping_chunks(
        &self,
        chunk_prunability: &ArrayData,
        begin: usize,
        end: usize,
    ) -> VortexResult<Prune> {
        let layouts = self
            .children_for_row_range(begin, end)
            .iter()
            .map(|x| *x as u64)
            .collect::<Vec<_>>();
        let chunks_prunable = take(chunk_prunability, ArrayData::from(layouts))?;

        if !chunks_prunable
            .statistics()
            .compute_as::<bool>(Stat::IsConstant)
            .vortex_expect("all boolean arrays must support is constant")
        {
            return Ok(Prune::CannotPrune);
        }

        // if the expression is constant null, this slice of chunks is not prunable
        let prunable = scalar_at(chunks_prunable, 0)?
            .as_bool()
            .value()
            .unwrap_or(false);
        Ok(if prunable {
            Prune::CanPrune
        } else {
            Prune::CannotPrune
        })
    }
}

impl LayoutReader for ChunkedLayoutReader {
    fn add_splits(&self, row_offset: usize, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
        for RangedLayoutReader((begin, _), child) in &self.layouts {
            child.add_splits(row_offset + begin, splits)?
        }
        Ok(())
    }

    fn poll_read(&self, selector: &RowMask) -> VortexResult<Option<PollRead<ArrayData>>> {
        let messages_to_fetch = self.buffer_read(selector)?;
        if !messages_to_fetch.is_empty() {
            return Ok(Some(PollRead::ReadMore(messages_to_fetch)));
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
                0 | 1 => Ok(child_arrays.pop().map(PollRead::Value)),
                _ => {
                    let dtype = child_arrays[0].dtype().clone();
                    Ok(Some(PollRead::Value(
                        ChunkedArray::try_new(child_arrays, dtype)?.into_array(),
                    )))
                }
            }
        } else {
            Ok(None)
        }
    }

    fn poll_metadata(&self) -> VortexResult<Option<PollRead<Vec<Option<ArrayData>>>>> {
        // Every chunked layout contains an optional "metadata" layout, which contains the
        // per-chunk statistics table.
        let Some(metadata_layout) = self.metadata_layout() else {
            return Ok(None);
        };

        if let Some(md) = self.cached_metadata.get() {
            return Ok(Some(PollRead::Value(vec![Some(md.clone())])));
        }

        match metadata_layout.poll_read(&RowMask::new_valid_between(0, self.n_chunks()))? {
            Some(PollRead::Value(array)) => {
                // We don't care if the write failed
                _ = self.cached_metadata.set(array.clone());
                Ok(Some(PollRead::Value(vec![Some(array)])))
            }
            Some(PollRead::ReadMore(messages)) => Ok(Some(PollRead::ReadMore(messages))),
            None => Ok(None),
        }
    }

    fn poll_prune(&self, begin: usize, end: usize) -> VortexResult<PollRead<Prune>> {
        if let Some(chunk_prunability) = self.cached_prunability.get() {
            return Ok(PollRead::Value(self.can_prune_overlapping_chunks(
                chunk_prunability,
                begin,
                end,
            )?));
        }

        let Some(predicate_expression) = self.scan.expr.as_ref() else {
            return Ok(PollRead::Value(Prune::CannotPrune));
        };

        if let Some(mr) = self.poll_metadata()? {
            Ok(match mr {
                PollRead::ReadMore(messages) => PollRead::ReadMore(messages),
                PollRead::Value(mut batches) => {
                    if batches.len() != 1 {
                        vortex_bail!("chunked layout should have exactly one metadata array");
                    }
                    let Some(metadata) = batches.swap_remove(0) else {
                        vortex_bail!("chunked layout should have exactly one metadata array")
                    };
                    let prunability = PruningPredicate::try_new(predicate_expression)
                        .map(|p| p.evaluate(&metadata))
                        .transpose()?
                        .flatten();

                    match prunability {
                        Some(chunk_prunability) => {
                            let is_selection_pruned =
                                self.can_prune_overlapping_chunks(&chunk_prunability, begin, end)?;
                            let _ = self.cached_prunability.set(chunk_prunability); // Losing the race is fine
                            PollRead::Value(is_selection_pruned)
                        }
                        None => PollRead::Value(Prune::CannotPrune),
                    }
                }
            })
        } else {
            Ok(PollRead::Value(Prune::CannotPrune))
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
    use flatbuffers::{root, FlatBufferBuilder};
    use futures_util::TryStreamExt;
    use vortex_array::array::{ChunkedArray, PrimitiveArray};
    use vortex_array::compute::FilterMask;
    use vortex_array::{ArrayDType, ArrayLen, IntoArrayData, IntoArrayVariant};
    use vortex_dtype::PType;
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
        // FIXME(ngates): impl From<LayoutSpec> for fb::Layout
        let chunked_layout = write::LayoutSpec::chunked(flat_layouts.into(), len as u64, None);
        let flat_buf = chunked_layout.write_flatbuffer(&mut fb);
        fb.finish_minimal(flat_buf);
        let fb_bytes = Bytes::copy_from_slice(fb.finished_data());
        let layout = root::<footer::Layout>(&fb_bytes).unwrap();

        let dtype = Arc::new(LazyDType::from_dtype(PType::I32.into()));
        let layout_builder = LayoutDeserializer::default();
        (
            ChunkedLayoutBuilder {
                layout,
                scan,
                layout_builder: layout_builder.clone(),
                message_cache: RelativeLayoutCache::new(cache.clone(), dtype.clone()),
            }
            .build()
            .unwrap(),
            ChunkedLayoutBuilder {
                layout,
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
            RowMask::try_new(FilterMask::from(first_range.finish()), 0, 200).unwrap(),
            RowMask::try_new(FilterMask::from(snd_range.finish()), 200, 400).unwrap(),
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
