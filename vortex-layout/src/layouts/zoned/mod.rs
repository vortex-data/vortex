//! Zoned layouts wrap a data layout with an auxiliary per-zone statistics layout.
//!
//! The zoned layout tree has exactly two children:
//! - a transparent `data` child containing the underlying column data
//! - an auxiliary `zones` child containing one row of aggregate statistics per zone
//!
//! Metadata stores the logical zone length in rows plus the aggregate functions present in the
//! auxiliary table. During scans, pruning first evaluates a falsification predicate against the
//! `zones` child and only forwards surviving rows to the underlying `data` child.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod builder;
mod pruning;
mod reader;
mod schema;
pub mod writer;
pub mod zone_map;

use std::sync::Arc;

pub(crate) use builder::AggregateStatsAccumulator;
pub(crate) use builder::aggregate_partials;
use prost::Message;
pub use schema::MAX_IS_TRUNCATED;
pub use schema::MIN_IS_TRUNCATED;
use vortex_array::DeserializeMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::aggregate_fn::AggregateFnRef;
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

use crate::LayoutBuildContext;
use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::children::OwnedLayoutChildren;
use crate::layouts::zoned::reader::ZonedReader;
use crate::layouts::zoned::schema::AggregateSpecProto;
use crate::layouts::zoned::schema::aggregate_fns_from_specs;
use crate::layouts::zoned::schema::aggregate_specs_from_fns;
use crate::layouts::zoned::schema::aggregate_stats_table_dtype;
use crate::layouts::zoned::schema::legacy_stats_table_dtype;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(Zoned);
vtable!(LegacyStats);

impl VTable for Zoned {
    type Layout = ZonedLayout;
    type Encoding = ZonedLayoutEncoding;
    type Metadata = ZonedMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new("vortex.zoned")
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
            aggregate_specs: match &layout.zone_map_schema {
                ZoneMapSchema::AggregateFns(aggregate_fns) => {
                    aggregate_specs_from_fns(aggregate_fns).vortex_expect(
                        "aggregate functions should be validated as serializable during build",
                    )
                }
                ZoneMapSchema::LegacyStats(_) => {
                    vortex_panic!("Cannot serialize legacy stats schema as vortex.zoned")
                }
            },
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
            1 => layout.children.child(1, &layout.stats_table_dtype),
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
        build_ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<Self::Layout> {
        vortex_ensure_eq!(
            children.nchildren(),
            2,
            "ZonedLayout expects exactly 2 children (data, zones)"
        );
        let aggregate_fns = aggregate_fns_from_specs(&metadata.aggregate_specs, build_ctx.session)?;
        aggregate_specs_from_fns(&aggregate_fns)?;
        let stats_table_dtype = aggregate_stats_table_dtype(dtype, &aggregate_fns);
        Ok(ZonedLayout {
            dtype: dtype.clone(),
            children: children.to_arc(),
            zone_len: metadata.zone_len as usize,
            zone_map_schema: ZoneMapSchema::AggregateFns(aggregate_fns),
            stats_table_dtype,
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

// TODO: This legacy vtable is only needed until layouts move onto the new vtable structure, where
// a LayoutPlugin can deserialize legacy `vortex.stats` metadata directly into `vortex.zoned`.
impl VTable for LegacyStats {
    type Layout = LegacyStatsLayout;
    type Encoding = LegacyStatsLayoutEncoding;
    type Metadata = LegacyStatsMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new("vortex.stats")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(LegacyStatsLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        <Zoned as VTable>::row_count(&layout.0)
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        <Zoned as VTable>::dtype(&layout.0)
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        LegacyStatsMetadata {
            zone_len: u32::try_from(layout.0.zone_len).vortex_expect("Invalid zone length"),
            zone_map_schema: layout.0.zone_map_schema.clone(),
        }
    }

    fn segment_ids(layout: &Self::Layout) -> Vec<SegmentId> {
        <Zoned as VTable>::segment_ids(&layout.0)
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        <Zoned as VTable>::nchildren(&layout.0)
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        <Zoned as VTable>::child(&layout.0, idx)
    }

    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType {
        <Zoned as VTable>::child_type(&layout.0, idx)
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &crate::LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ZonedReader::try_new(
            layout.0.clone(),
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
        metadata: &LegacyStatsMetadata,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _build_ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<Self::Layout> {
        vortex_ensure_eq!(
            children.nchildren(),
            2,
            "LegacyStatsLayout expects exactly 2 children (data, zones)"
        );
        let stats_table_dtype = match &metadata.zone_map_schema {
            ZoneMapSchema::LegacyStats(stats) => legacy_stats_table_dtype(dtype, stats),
            ZoneMapSchema::AggregateFns(aggregate_fns) => {
                aggregate_stats_table_dtype(dtype, aggregate_fns)
            }
        };
        Ok(LegacyStatsLayout(ZonedLayout {
            dtype: dtype.clone(),
            children: children.to_arc(),
            zone_len: metadata.zone_len as usize,
            zone_map_schema: metadata.zone_map_schema.clone(),
            stats_table_dtype,
        }))
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        <Zoned as VTable>::with_children(&mut layout.0, children)
    }
}

/// Encoding marker for the zoned layout.
#[derive(Debug)]
pub struct ZonedLayoutEncoding;

/// Encoding marker for the legacy `vortex.stats` zoned layout.
#[derive(Debug)]
pub struct LegacyStatsLayoutEncoding;

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
    zone_map_schema: ZoneMapSchema,
    stats_table_dtype: DType,
}

/// A legacy `vortex.stats` layout backed by the shared zoned runtime implementation.
#[derive(Clone, Debug)]
pub struct LegacyStatsLayout(ZonedLayout);

impl LegacyStatsLayout {
    /// Returns display names for the zone-map aggregates stored by this layout.
    pub fn present_aggregates(&self) -> Arc<[String]> {
        self.0.present_aggregates()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ZoneMapSchema {
    LegacyStats(Arc<[Stat]>),
    AggregateFns(Arc<[AggregateFnRef]>),
}

impl ZonedLayout {
    /// Create a zoned layout from a data child, a zone-map child, a zone length, and the aggregate
    /// functions stored in the zone map.
    pub fn try_new(
        data: LayoutRef,
        zones: LayoutRef,
        zone_len: usize,
        aggregate_fns: Arc<[AggregateFnRef]>,
    ) -> VortexResult<Self> {
        vortex_ensure!(zone_len > 0, "Zone length must be greater than 0");

        let expected_dtype = aggregate_stats_table_dtype(data.dtype(), &aggregate_fns);
        if zones.dtype() != &expected_dtype {
            vortex_bail!("Invalid zone map layout: zones dtype does not match expected dtype");
        }
        aggregate_specs_from_fns(&aggregate_fns)?;

        Ok(Self {
            dtype: data.dtype().clone(),
            children: OwnedLayoutChildren::layout_children(vec![data, zones]),
            zone_len,
            zone_map_schema: ZoneMapSchema::AggregateFns(aggregate_fns),
            stats_table_dtype: expected_dtype,
        })
    }

    pub fn nzones(&self) -> usize {
        usize::try_from(self.children.child_row_count(1)).vortex_expect("Invalid number of zones")
    }

    pub fn zone_len(&self) -> usize {
        self.zone_len
    }

    /// Returns display names for the zone-map aggregates stored by this layout.
    pub fn present_aggregates(&self) -> Arc<[String]> {
        match &self.zone_map_schema {
            ZoneMapSchema::LegacyStats(stats) => stats
                .iter()
                .filter_map(Stat::aggregate_fn)
                .map(|aggregate_fn| aggregate_fn.to_string())
                .collect::<Vec<_>>()
                .into(),
            ZoneMapSchema::AggregateFns(aggregate_fns) => aggregate_fns
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .into(),
        }
    }

    pub(super) fn aggregate_fns(
        &self,
        _session: &VortexSession,
    ) -> VortexResult<Arc<[AggregateFnRef]>> {
        match &self.zone_map_schema {
            ZoneMapSchema::LegacyStats(stats) => Ok(stats
                .iter()
                .filter_map(Stat::aggregate_fn)
                .collect::<Vec<_>>()
                .into()),
            ZoneMapSchema::AggregateFns(aggregate_fns) => Ok(Arc::clone(aggregate_fns)),
        }
    }

    pub(super) fn stats_table_dtype_for(&self, aggregate_fns: &[AggregateFnRef]) -> DType {
        if let ZoneMapSchema::LegacyStats(stats) = &self.zone_map_schema {
            return legacy_stats_table_dtype(&self.dtype, stats);
        }

        aggregate_stats_table_dtype(&self.dtype, aggregate_fns)
    }
}

/// Serialized zoned-layout metadata.
///
/// `zone_len` is the logical row length of each zone. `aggregate_specs` is the ordered list of
/// aggregate functions stored in the auxiliary stats-table child.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ZonedMetadata {
    pub(super) zone_len: u32,
    pub(super) aggregate_specs: Arc<[AggregateSpecProto]>,
}

/// Serialized metadata for legacy `vortex.stats` layouts.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LegacyStatsMetadata {
    pub(super) zone_len: u32,
    pub(crate) zone_map_schema: ZoneMapSchema,
}

const ZONED_METADATA_PROTO_VERSION: u8 = 1;

#[derive(Clone, PartialEq, Message)]
struct ZonedMetadataProto {
    #[prost(uint32, tag = "1")]
    zone_len: u32,
    #[prost(message, repeated, tag = "2")]
    aggregate_specs: Vec<AggregateSpecProto>,
}

impl DeserializeMetadata for ZonedMetadata {
    type Output = Self;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        let Some((&version, proto_bytes)) = metadata.split_first() else {
            vortex_bail!("Zoned metadata missing protobuf version");
        };

        vortex_ensure!(
            version == ZONED_METADATA_PROTO_VERSION,
            "Unsupported zoned metadata version: {}",
            version
        );
        vortex_ensure!(!proto_bytes.is_empty(), "Zoned metadata missing protobuf");

        let proto = ZonedMetadataProto::decode(proto_bytes)?;
        Ok(Self {
            zone_len: proto.zone_len,
            aggregate_specs: proto.aggregate_specs.into(),
        })
    }
}

impl SerializeMetadata for ZonedMetadata {
    fn serialize(self) -> Vec<u8> {
        let proto = ZonedMetadataProto {
            zone_len: self.zone_len,
            aggregate_specs: self.aggregate_specs.to_vec(),
        };
        let mut metadata = vec![ZONED_METADATA_PROTO_VERSION];
        metadata.extend(proto.encode_to_vec());
        metadata
    }
}

impl DeserializeMetadata for LegacyStatsMetadata {
    type Output = Self;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        vortex_ensure!(
            metadata.len() >= 4,
            "Legacy zoned metadata must contain at least 4 bytes for zone length, got {}",
            metadata.len()
        );

        // Backward compat: older files may encode `zone_len == 0`. Preserve the raw metadata on
        // read and let the reader disable zoned pruning for those layouts instead of rejecting
        // deserialization outright.
        let zone_len = u32::try_from_le_bytes(&metadata[0..4])?;
        let present_stats: Arc<[Stat]> = stats_from_bitset_bytes(&metadata[4..]).into();

        Ok(Self {
            zone_len,
            zone_map_schema: ZoneMapSchema::LegacyStats(present_stats),
        })
    }
}

impl SerializeMetadata for LegacyStatsMetadata {
    fn serialize(self) -> Vec<u8> {
        match self.zone_map_schema {
            ZoneMapSchema::LegacyStats(stats) => {
                let mut metadata = self.zone_len.to_le_bytes().to_vec();
                metadata.extend(as_stat_bitset_bytes(&stats));
                metadata
            }
            ZoneMapSchema::AggregateFns(_) => {
                vortex_panic!("Cannot serialize aggregate specs as legacy stats metadata")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic;

    use rstest::rstest;
    use vortex_array::aggregate_fn::AggregateFnRef;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
    use vortex_array::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::session::AggregateFnSession;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::stats::as_stat_bitset_bytes;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::IntoLayout;
    use crate::children::OwnedLayoutChildren;
    use crate::layouts::flat::FlatLayout;
    use crate::segments::SegmentId;

    fn aggregate_spec(aggregate_fn: AggregateFnRef) -> AggregateSpecProto {
        AggregateSpecProto::try_from_aggregate_fn(&aggregate_fn).unwrap()
    }

    #[rstest]
    #[case(ZonedMetadata {
            zone_len: u32::MAX,
            aggregate_specs: Arc::new([]),
        })]
    #[case::min_max(ZonedMetadata {
            zone_len: 314,
            aggregate_specs: Arc::new([
                aggregate_spec(Max.bind(NumericalAggregateOpts::skip_nans())),
                aggregate_spec(Min.bind(NumericalAggregateOpts::skip_nans())),
            ]),
        })]
    fn test_metadata_serialization(#[case] metadata: ZonedMetadata) {
        let serialized = metadata.clone().serialize();
        assert_eq!(serialized[0], ZONED_METADATA_PROTO_VERSION);
        let deserialized = ZonedMetadata::deserialize(&serialized).unwrap();
        assert_eq!(deserialized, metadata);
    }

    #[test]
    fn test_metadata_serialization_preserves_aggregate_options() -> VortexResult<()> {
        let aggregate_fn = BoundedMax.bind(BoundedMaxOptions {
            // SAFETY: 128 is non-zero.
            max_bytes: unsafe { std::num::NonZeroUsize::new_unchecked(128) },
        });
        let metadata = ZonedMetadata {
            zone_len: 314,
            aggregate_specs: Arc::new([AggregateSpecProto::try_from_aggregate_fn(&aggregate_fn)?]),
        };

        let deserialized = ZonedMetadata::deserialize(&metadata.serialize())?;
        let session = VortexSession::empty().with::<AggregateFnSession>();
        let aggregate_fns = aggregate_fns_from_specs(&deserialized.aggregate_specs, &session)?;

        assert_eq!(aggregate_fns.as_ref(), std::slice::from_ref(&aggregate_fn));
        Ok(())
    }

    #[test]
    fn test_deserialize_legacy_stat_bitset_as_legacy_stats() {
        let mut serialized = u32::MAX.to_le_bytes().to_vec();
        serialized.extend(as_stat_bitset_bytes(&[
            Stat::IsStrictSorted,
            Stat::IsSorted,
            Stat::Max,
        ]));
        let deserialized = LegacyStatsMetadata::deserialize(&serialized).unwrap();
        let ZoneMapSchema::LegacyStats(legacy_stats) = deserialized.zone_map_schema else {
            panic!("legacy bitset metadata should deserialize as legacy stats");
        };

        assert!(legacy_stats.is_sorted());
        assert_eq!(
            legacy_stats.as_ref(),
            &[Stat::IsSorted, Stat::IsStrictSorted, Stat::Max]
        );
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::unsupported_version(vec![0])]
    #[case::missing_proto(vec![ZONED_METADATA_PROTO_VERSION])]
    #[case::malformed_proto(vec![ZONED_METADATA_PROTO_VERSION, 0])]
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
        let deserialized = LegacyStatsMetadata::deserialize(&metadata).unwrap();
        assert_eq!(deserialized.zone_len, 0);
        let ZoneMapSchema::LegacyStats(legacy_stats) = deserialized.zone_map_schema else {
            panic!("legacy bitset metadata should deserialize as legacy stats");
        };
        assert!(legacy_stats.is_empty());
    }

    #[test]
    fn test_build_allows_zero_zone_len_for_backcompat() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let read_ctx = ReadContext::new([]);
        let children = OwnedLayoutChildren::layout_children(vec![
            FlatLayout::new(0, dtype.clone(), SegmentId::from(0), read_ctx.clone()).into_layout(),
            FlatLayout::new(
                0,
                legacy_stats_table_dtype(&dtype, &[]),
                SegmentId::from(1),
                read_ctx,
            )
            .into_layout(),
        ]);
        let session = vortex_array::array_session();
        let build_read_ctx = ReadContext::new([]);
        let build_ctx = LayoutBuildContext {
            session: &session,
            array_read_ctx: &build_read_ctx,
        };

        let layout = <LegacyStats as VTable>::build(
            &LegacyStatsLayoutEncoding,
            &dtype,
            0,
            &LegacyStatsMetadata {
                zone_len: 0,
                zone_map_schema: ZoneMapSchema::LegacyStats(Arc::new([])),
            },
            vec![],
            children.as_ref(),
            &build_ctx,
        )?;

        assert_eq!(layout.0.zone_len, 0);
        Ok(())
    }

    #[test]
    fn test_build_rejects_invalid_child_count() {
        let metadata = ZonedMetadata {
            zone_len: 3,
            aggregate_specs: Arc::new([]),
        };
        let children = OwnedLayoutChildren::layout_children(vec![]);
        let session = vortex_array::array_session();
        let build_read_ctx = ReadContext::new([]);
        let build_ctx = LayoutBuildContext {
            session: &session,
            array_read_ctx: &build_read_ctx,
        };

        let result = <Zoned as VTable>::build(
            &ZonedLayoutEncoding,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
            0,
            &metadata,
            vec![],
            children.as_ref(),
            &build_ctx,
        );

        assert!(result.is_err());
    }
}
