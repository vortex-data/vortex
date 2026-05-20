//! Zoned layouts wrap a data layout with an auxiliary per-zone statistics layout.
//!
//! The zoned layout tree has exactly two children:
//! - a transparent `data` child containing the underlying column data
//! - an auxiliary `zones` child containing one row of aggregate statistics per zone
//!
//! Metadata stores the logical zone length in rows plus the sorted list of statistics present in
//! the auxiliary table. During scans, pruning first evaluates a falsification predicate against
//! the `zones` child and only forwards surviving rows to the underlying `data` child.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod builder;
mod pruning;
mod reader;
mod schema;
pub mod writer;
pub mod zone_map;

use std::sync::Arc;

pub(crate) use builder::StatsAccumulator;
pub use schema::MAX_IS_TRUNCATED;
pub use schema::MIN_IS_TRUNCATED;
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
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
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
use crate::layouts::zoned::schema::stats_table_dtype;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(Zoned);

impl VTable for Zoned {
    type Layout = ZonedLayout;
    type Encoding = ZonedLayoutEncoding;
    type Metadata = ZonedMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        // For legacy reasons the serialized layout encoding ID is still `vortex.stats`.
        LayoutId::new("vortex.stats")
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
            1 => layout
                .children
                .child(1, &stats_table_dtype(layout.dtype(), &layout.present_stats)),
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
        ctx: &crate::LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ZonedReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
            ctx.clone(),
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
        vortex_ensure_eq!(
            children.nchildren(),
            2,
            "ZonedLayout expects exactly 2 children (data, zones)"
        );
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

/// Encoding marker for the zoned layout.
#[derive(Debug)]
pub struct ZonedLayoutEncoding;

/// A layout that annotates a data child with one row of aggregate statistics per zone.
///
/// The first child is the underlying data layout. The second child is an auxiliary stats table
/// whose rows align with logical row zones of length `zone_len`, except for the final partial zone.
/// During reads, pruning uses the stats table to skip zones whose rows cannot satisfy a filter.
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
        let expected_dtype = stats_table_dtype(data.dtype(), &present_stats);
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

    pub fn zone_len(&self) -> usize {
        self.zone_len
    }

    /// Returns an array of stats that exist in the layout's data, must be sorted.
    pub fn present_stats(&self) -> &Arc<[Stat]> {
        &self.present_stats
    }
}

/// Serialized zoned-layout metadata.
///
/// `zone_len` is the logical row length of each zone. `present_stats` is the sorted list of
/// statistics stored in the auxiliary stats-table child.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ZonedMetadata {
    pub(super) zone_len: u32,
    pub(super) present_stats: Arc<[Stat]>,
}

impl DeserializeMetadata for ZonedMetadata {
    type Output = Self;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        vortex_ensure!(
            metadata.len() >= 4,
            "Zoned metadata must contain at least 4 bytes for zone length, got {}",
            metadata.len()
        );

        // Backward compat: older files may encode `zone_len == 0`. Preserve the raw metadata on
        // read and let the reader disable zoned pruning for those layouts instead of rejecting
        // deserialization outright.
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
    use std::panic;

    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::IntoLayout;
    use crate::children::OwnedLayoutChildren;
    use crate::layouts::flat::FlatLayout;
    use crate::segments::SegmentId;

    #[rstest]
    #[case(ZonedMetadata {
            zone_len: u32::MAX,
            present_stats: Arc::new([]),
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

    #[rstest]
    #[case(vec![])]
    #[case(vec![0])]
    #[case(vec![0, 0])]
    #[case(vec![0, 0, 0])]
    fn test_deserialize_short_metadata_errors(#[case] metadata: Vec<u8>) {
        assert!(ZonedMetadata::deserialize(&metadata).is_err());
    }

    #[test]
    fn test_deserialize_short_metadata_returns_error_not_panic() {
        let result = panic::catch_unwind(|| ZonedMetadata::deserialize(&[]));
        assert!(
            result.is_ok(),
            "deserialize should return an error, not panic"
        );
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn test_deserialize_zero_zone_len_is_allowed_for_backcompat() {
        let metadata = 0u32.to_le_bytes();
        let deserialized = ZonedMetadata::deserialize(&metadata).unwrap();
        assert_eq!(deserialized.zone_len, 0);
        assert!(deserialized.present_stats.is_empty());
    }

    #[test]
    fn test_build_allows_zero_zone_len_for_backcompat() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let read_ctx = ReadContext::new([]);
        let children = OwnedLayoutChildren::layout_children(vec![
            FlatLayout::new(0, dtype.clone(), SegmentId::from(0), read_ctx.clone()).into_layout(),
            FlatLayout::new(
                0,
                stats_table_dtype(&dtype, &[]),
                SegmentId::from(1),
                read_ctx,
            )
            .into_layout(),
        ]);

        let layout = <Zoned as VTable>::build(
            &ZonedLayoutEncoding,
            &dtype,
            0,
            &ZonedMetadata {
                zone_len: 0,
                present_stats: Arc::new([]),
            },
            vec![],
            children.as_ref(),
            &ReadContext::new([]),
        )?;

        assert_eq!(layout.zone_len, 0);
        Ok(())
    }

    #[test]
    fn test_build_rejects_invalid_child_count() {
        let metadata = ZonedMetadata {
            zone_len: 3,
            present_stats: Arc::new([]),
        };
        let children = OwnedLayoutChildren::layout_children(vec![]);

        let result = <Zoned as VTable>::build(
            &ZonedLayoutEncoding,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
            0,
            &metadata,
            vec![],
            children.as_ref(),
            &ReadContext::new([]),
        );

        assert!(result.is_err());
    }
}
