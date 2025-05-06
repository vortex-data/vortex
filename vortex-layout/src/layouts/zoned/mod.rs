mod builder;
mod reader;
pub mod writer;
pub mod zone_map;

use std::collections::BTreeSet;
use std::sync::Arc;

pub use builder::{MAX_IS_TRUNCATED, MIN_IS_TRUNCATED, lower_bound, upper_bound};
use vortex_array::stats::{Stat, as_stat_bitset_bytes, stats_from_bitset_bytes};
use vortex_array::{ArrayContext, DeserializeMetadata, SerializeMetadata};
use vortex_dtype::{DType, FieldMask, TryFromBytes};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};

use crate::children::LayoutChildren;
use crate::layouts::zoned::reader::ZonedReader;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildType, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutRef, VTable, vtable,
};

vtable!(Zoned);

impl VTable for ZonedVTable {
    type Layout = ZonedLayout;
    type Encoding = ZonedLayoutEncoding;
    type Metadata = ZonedMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("vortex.stats") // For legacy reasons, this is called stats
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(ZonedLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.data.row_count()
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        layout.data.dtype()
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        ZonedMetadata {
            zone_len: u32::try_from(layout.zone_len).vortex_expect("Invalid zone length"),
            present_stats: layout.present_stats.clone(),
        }
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        2
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        match idx {
            0 => Ok(layout.data.clone()),
            1 => Ok(layout.zones.clone()),
            _ => vortex_bail!("Invalid child index: {}", idx),
        }
    }

    fn child_type(_layout: &Self::Layout, idx: usize) -> LayoutChildType {
        match idx {
            0 => LayoutChildType::Transparent("data".into()),
            1 => LayoutChildType::Auxiliary("zones".into()),
            _ => vortex_panic!("Invalid child index: {}", idx),
        }
    }

    fn register_splits(
        layout: &Self::Layout,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        layout.data.register_splits(field_mask, row_offset, splits)
    }

    fn new_reader(
        layout: &Self::Layout,
        name: &Arc<str>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ZonedReader::try_new(
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
        let data = children.child(0, dtype)?;

        let zones_dtype = ZoneMap::dtype_for_stats_table(data.dtype(), &metadata.present_stats);
        let zones = children.child(1, &zones_dtype)?;

        Ok(ZonedLayout::new(
            data,
            zones,
            metadata.zone_len as usize,
            metadata.present_stats.clone(),
        ))
    }
}

#[derive(Debug)]
pub struct ZonedLayoutEncoding;

#[derive(Clone, Debug)]
pub struct ZonedLayout {
    data: LayoutRef,
    zones: LayoutRef,
    zone_len: usize,
    present_stats: Arc<[Stat]>,
}

impl ZonedLayout {
    pub fn new(
        data: LayoutRef,
        zones: LayoutRef,
        zone_len: usize,
        present_stats: Arc<[Stat]>,
    ) -> Self {
        let expected_dtype = ZoneMap::dtype_for_stats_table(data.dtype(), &present_stats);
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

    pub fn nzones(&self) -> usize {
        usize::try_from(self.zones.row_count()).vortex_expect("Invalid number of zones")
    }

    pub fn present_stats(&self) -> &Arc<[Stat]> {
        &self.present_stats
    }
}

#[derive(Debug)]
pub struct ZonedMetadata {
    pub(super) zone_len: u32,
    pub(super) present_stats: Arc<[Stat]>,
}

impl DeserializeMetadata for ZonedMetadata {
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

impl SerializeMetadata for ZonedMetadata {
    fn serialize(self) -> Vec<u8> {
        let mut metadata = vec![];
        // First, write the block size to the metadata.
        metadata.extend_from_slice(&self.zone_len.to_le_bytes());
        // Then write the bit-set of statistics.
        metadata.extend_from_slice(&as_stat_bitset_bytes(&self.present_stats));
        metadata
    }
}
