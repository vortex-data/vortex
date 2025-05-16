mod reader;
pub mod stats_table;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::stats::{Stat, as_stat_bitset_bytes, stats_from_bitset_bytes};
use vortex_array::{ArrayContext, DeserializeMetadata, SerializeMetadata};
use vortex_dtype::{DType, FieldMask, FieldPath, TryFromBytes};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};

use crate::layouts::stats::reader::ZoneMapReader;
use crate::layouts::stats::stats_table::StatsTable;
use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildren, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutRef, LayoutVisitor, VTable,
    vtable,
};

vtable!(ZoneMap);

impl VTable for ZoneMapVTable {
    type Layout = ZoneMapLayout;
    type Encoding = ZoneMapLayoutEncoding;
    type Metadata = ZoneMapMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("vortex.stats") // For legacy reasons, this is called stats
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(ZoneMapLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.data.row_count()
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        layout.data.dtype()
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        2
    }

    fn visit_children(
        layout: &Self::Layout,
        _field_mask: Option<&[FieldMask]>,
        visitor: &mut dyn LayoutVisitor,
    ) {
        visitor.visit_child("data", 0, Some(&FieldPath::root()), &layout.data);
        visitor.visit_child("zones", 0, None, &layout.zones);
    }

    fn register_splits(
        layout: &Self::Layout,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) {
        layout.data.register_splits(field_mask, row_offset, splits)
    }

    fn new_reader(
        layout: &Self::Layout,
        name: &Arc<str>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ZoneMapReader::try_new(
            layout.clone(),
            name.clone(),
            segment_source.clone(),
            ctx.clone(),
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        _row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
    ) -> VortexResult<Self::Layout> {
        let data = children.child(0, dtype);

        let zones_dtype = StatsTable::dtype_for_stats_table(data.dtype(), &metadata.present_stats);
        let zones = children.child(1, &zones_dtype);

        Ok(ZoneMapLayout::new(
            data,
            zones,
            metadata.zone_len as usize,
            metadata.present_stats.clone(),
        ))
    }
}

#[derive(Debug)]
pub struct ZoneMapLayoutEncoding;

#[derive(Clone)]
pub struct ZoneMapLayout {
    data: LayoutRef,
    zones: LayoutRef,
    zone_len: usize,
    present_stats: Arc<[Stat]>,
}

impl ZoneMapLayout {
    pub fn new(
        data: LayoutRef,
        zones: LayoutRef,
        zone_len: usize,
        present_stats: Arc<[Stat]>,
    ) -> Self {
        let expected_dtype = StatsTable::dtype_for_stats_table(data.dtype(), &present_stats);
        if zones.dtype() != &expected_dtype {
            vortex_panic!("Invalid zone map layout: zones dtype does not match expected dtype");
        }
        Self {
            data,
            zones,
            zone_len,
            present_stats,
        }
    }

    pub(super) fn nzones(&self) -> usize {
        usize::try_from(self.zones.row_count()).vortex_expect("Invalid number of zones")
    }
}

#[derive(Debug)]
pub struct ZoneMapMetadata {
    pub(super) zone_len: u32,
    pub(super) present_stats: Arc<[Stat]>,
}

impl DeserializeMetadata for ZoneMapMetadata {
    type Output = Self;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        let zone_len = u32::try_from_le_bytes(&metadata[0..4])?;
        let present_stats: Arc<[Stat]> = stats_from_bitset_bytes(&metadata[4..]).into();
        Ok(Self {
            zone_len,
            present_stats,
        })
    }
}

impl SerializeMetadata for ZoneMapMetadata {
    fn serialize(self) -> Vec<u8> {
        let mut metadata = vec![];
        // First, write the block size to the metadata.
        metadata.extend_from_slice(&self.zone_len.to_le_bytes());
        // Then write the bit-set of statistics.
        metadata.extend_from_slice(&as_stat_bitset_bytes(&self.present_stats));
        metadata
    }
}
