// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod builder;
mod reader;
pub mod writer;
pub mod zone_map;

use std::sync::Arc;

pub use builder::MAX_IS_TRUNCATED;
pub use builder::MIN_IS_TRUNCATED;
use vortex_array::DeserializeMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::TryFromBytes;
use vortex_array::expr::stats::Stat;
use vortex_array::stats::as_stat_bitset_bytes;
use vortex_array::stats::stats_from_bitset_bytes;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::children::OwnedLayoutChildren;
use crate::layouts::zoned::reader::ZonedReader;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(Zoned);

impl VTable for Zoned {
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
        layout.children.child_row_count(0)
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        ZonedMetadata {
            zone_len: u32::try_from(layout.zone_len).vortex_expect("Invalid zone length"),
            present_stats: Arc::clone(&layout.present_stats),
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
            0 => layout.children.child(0, layout.dtype()),
            1 => layout.children.child(
                1,
                &ZoneMap::dtype_for_stats_table(layout.dtype(), &layout.present_stats),
            ),
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

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ZonedReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        _row_count: u64,
        metadata: &ZonedMetadata,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: &ReadContext,
    ) -> VortexResult<Self::Layout> {
        Ok(ZonedLayout {
            dtype: dtype.clone(),
            children: children.to_arc(),
            zone_len: metadata.zone_len as usize,
            present_stats: Arc::clone(&metadata.present_stats),
        })
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        if children.len() != 2 {
            vortex_bail!(
                "ZonedLayout expects exactly 2 children (data, zones), got {}",
                children.len()
            );
        }
        layout.children = OwnedLayoutChildren::layout_children(children);
        Ok(())
    }
}

#[derive(Debug)]
pub struct ZonedLayoutEncoding;

/// Annotates a data layout with per-zone aggregate statistics (e.g. min, max, null count).
///
/// During reads, zone maps allow entire zones to be skipped when a filter predicate cannot match.
#[derive(Clone, Debug)]
pub struct ZonedLayout {
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
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
        if zone_len == 0 {
            vortex_panic!("Zone length must be greater than 0");
        }
        let expected_dtype = ZoneMap::dtype_for_stats_table(data.dtype(), &present_stats);
        if zones.dtype() != &expected_dtype {
            vortex_panic!("Invalid zone map layout: zones dtype does not match expected dtype");
        }
        Self {
            dtype: data.dtype().clone(),
            children: OwnedLayoutChildren::layout_children(vec![data, zones]),
            zone_len,
            present_stats,
        }
    }

    pub fn nzones(&self) -> usize {
        usize::try_from(self.children.child_row_count(1)).vortex_expect("Invalid number of zones")
    }

    /// Returns an array of stats that exist in the layout's data, must be sorted.
    pub fn present_stats(&self) -> &Arc<[Stat]> {
        &self.present_stats
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(ZonedMetadata {
            zone_len: u32::MAX,
            present_stats: Arc::new([]),
        })]
    #[case(ZonedMetadata {
            zone_len: 0,
            present_stats: Arc::new([Stat::IsConstant]),
        })]
    #[case::all_sorted(ZonedMetadata {
            zone_len: 314,
            present_stats: Arc::new([Stat::IsConstant, Stat::IsSorted, Stat::IsStrictSorted, Stat::Max, Stat::Min, Stat::Sum, Stat::NullCount, Stat::UncompressedSizeInBytes, Stat::NaNCount]),
        })]
    #[case::some_sorted(ZonedMetadata {
            zone_len: 314,
            present_stats: Arc::new([Stat::IsSorted, Stat::IsStrictSorted, Stat::Max, Stat::Min, Stat::Sum, Stat::NullCount, Stat::UncompressedSizeInBytes, Stat::NaNCount]),
        })]
    fn test_metadata_serialization(#[case] metadata: ZonedMetadata) {
        let serialized = metadata.clone().serialize();
        let deserialized = ZonedMetadata::deserialize(&serialized).unwrap();
        assert_eq!(deserialized, metadata);
    }

    #[test]
    fn test_deserialize_unsorted_stats() {
        let metadata = ZonedMetadata {
            zone_len: u32::MAX,
            present_stats: Arc::new([Stat::IsStrictSorted, Stat::IsSorted]),
        };
        let serialized = metadata.clone().serialize();
        let deserialized = ZonedMetadata::deserialize(&serialized).unwrap();
        assert!(deserialized.present_stats.is_sorted());
        assert_eq!(
            deserialized.present_stats.len(),
            metadata.present_stats.len()
        );
        assert_ne!(deserialized.present_stats, metadata.present_stats);
    }
}
