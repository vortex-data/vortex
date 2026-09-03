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

use std::num::NonZeroUsize;
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
use vortex_session::registry::CachedId;

use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::layouts::zoned::reader::ZonedReader;
use crate::layouts::zoned::schema::AggregateSpecProto;
use crate::layouts::zoned::schema::aggregate_specs_from_fns;
use crate::layouts::zoned::schema::aggregate_stats_table_dtype;
use crate::layouts::zoned::schema::legacy_stats_table_dtype;
use crate::layouts::zoned::schema::try_aggregate_fns_from_specs;
use crate::segments::SegmentSource;

/// Zoned layout vtable.
#[derive(Clone, Debug)]
pub struct Zoned;

/// Legacy `vortex.stats` layout vtable.
#[derive(Clone, Debug)]
pub struct LegacyStats;

/// Backwards-compatible plugin names.
pub use LegacyStats as LegacyStatsLayoutEncoding;
pub use Zoned as ZonedLayoutEncoding;

/// Zoned-layout-specific data.
#[derive(Clone, Debug)]
pub struct ZonedData {
    zone_len: usize,
    zone_map_schema: ZoneMapSchema,
    stats_table_dtype: DType,
}

/// A layout annotating a data child with per-zone statistics.
pub type ZonedLayout = Layout<Zoned>;
/// A legacy `vortex.stats` layout using the same runtime data.
pub type LegacyStatsLayout = Layout<LegacyStats>;

impl VTable for Zoned {
    type LayoutData = ZonedData;
    type Metadata = ZonedMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.zoned");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
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

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &ZonedMetadata,
    ) -> VortexResult<Self::LayoutData> {
        vortex_ensure_eq!(
            args.children.nchildren(),
            2,
            "ZonedLayout expects exactly 2 children (data, zones)"
        );
        vortex_ensure_eq!(args.children.child_row_count(0), args.row_count);
        let Some(aggregate_fns) =
            try_aggregate_fns_from_specs(&metadata.aggregate_specs, args.session)?
        else {
            args.children.child(0, args.dtype)?;
            return Ok(ZonedData {
                zone_len: 0,
                zone_map_schema: ZoneMapSchema::AggregateFns(Arc::new([])),
                stats_table_dtype: aggregate_stats_table_dtype(args.dtype, &[]),
            });
        };
        aggregate_specs_from_fns(&aggregate_fns)?;
        let stats_table_dtype = aggregate_stats_table_dtype(args.dtype, &aggregate_fns);
        args.children.child(0, args.dtype)?;
        args.children.child(1, &stats_table_dtype)?;
        Ok(ZonedData {
            zone_len: metadata.zone_len as usize,
            zone_map_schema: ZoneMapSchema::AggregateFns(aggregate_fns),
            stats_table_dtype,
        })
    }

    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            0 => Ok(layout.dtype().clone()),
            1 => Ok(layout.stats_table_dtype.clone()),
            _ => vortex_bail!("Invalid child index: {idx}"),
        }
    }

    fn child_type(_layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        match idx {
            0 => LayoutChildType::Transparent("data".into()),
            1 => LayoutChildType::Auxiliary("zones".into()),
            _ => vortex_panic!("Invalid child index: {}", idx),
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        layout.zoned_reader(name, segment_source, session, ctx)
    }
}

impl VTable for LegacyStats {
    type LayoutData = ZonedData;
    type Metadata = LegacyStatsMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.stats");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        LegacyStatsMetadata {
            zone_len: u32::try_from(layout.zone_len).vortex_expect("Invalid zone length"),
            zone_map_schema: layout.zone_map_schema.clone(),
        }
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &LegacyStatsMetadata,
    ) -> VortexResult<Self::LayoutData> {
        vortex_ensure_eq!(
            args.children.nchildren(),
            2,
            "LegacyStatsLayout expects exactly 2 children (data, zones)"
        );
        let stats_table_dtype = match &metadata.zone_map_schema {
            ZoneMapSchema::LegacyStats(stats) => legacy_stats_table_dtype(args.dtype, stats),
            ZoneMapSchema::AggregateFns(aggregate_fns) => {
                aggregate_stats_table_dtype(args.dtype, aggregate_fns)
            }
        };
        args.children.child(0, args.dtype)?;
        args.children.child(1, &stats_table_dtype)?;
        Ok(ZonedData {
            zone_len: metadata.zone_len as usize,
            zone_map_schema: metadata.zone_map_schema.clone(),
            stats_table_dtype,
        })
    }

    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            0 => Ok(layout.dtype().clone()),
            1 => Ok(layout.stats_table_dtype.clone()),
            _ => vortex_bail!("Invalid child index: {idx}"),
        }
    }

    fn child_type(_layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        match idx {
            0 => LayoutChildType::Transparent("data".into()),
            1 => LayoutChildType::Auxiliary("zones".into()),
            _ => vortex_panic!("Invalid child index: {idx}"),
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        layout.zoned_reader(name, segment_source, session, ctx)
    }
}

impl LegacyStatsLayout {
    fn zoned_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        if self.zone_len == 0 {
            return self
                .slot(0)?
                .vortex_expect("ZonedLayout always has a data child")
                .new_reader(name, segment_source, session, ctx);
        }

        Ok(Arc::new(ZonedReader::try_new(
            self.clone(),
            self.nzones(),
            name,
            segment_source,
            session.clone(),
            ctx.clone(),
        )?))
    }

    pub fn nzones(&self) -> usize {
        usize::try_from(self.children().child_row_count(1))
            .vortex_expect("Invalid number of zones, cannot handle more than usize zones")
    }

    /// Returns display names for the zone-map aggregates stored by this layout.
    pub fn present_aggregates(&self) -> Arc<[String]> {
        present_aggregates(&self.zone_map_schema)
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
        zone_len: NonZeroUsize,
        aggregate_fns: Arc<[AggregateFnRef]>,
    ) -> VortexResult<Self> {
        let expected_dtype = aggregate_stats_table_dtype(data.dtype(), &aggregate_fns);
        if zones.dtype() != &expected_dtype {
            vortex_bail!("Invalid zone map layout: zones dtype does not match expected dtype");
        }

        // Verify that every aggregate is serializable.
        aggregate_specs_from_fns(&aggregate_fns)?;

        let dtype = data.dtype().clone();
        let row_count = data.row_count();
        Ok(LayoutParts::new(
            Zoned,
            dtype,
            row_count,
            Vec::new(),
            OwnedLayoutChildren::layout_children(vec![data, zones]),
            ZonedData {
                zone_len: zone_len.get(),
                zone_map_schema: ZoneMapSchema::AggregateFns(aggregate_fns),
                stats_table_dtype: expected_dtype,
            },
        )
        .into_typed())
    }

    pub fn zone_len(&self) -> usize {
        self.zone_len
    }

    /// Returns display names for the zone-map aggregates stored by this layout.
    pub fn present_aggregates(&self) -> Arc<[String]> {
        present_aggregates(&self.zone_map_schema)
    }

    /// Builds a reader for a zoned layout, bypassing [`ZonedReader`] when the zone map is empty
    /// (`zone_len == 0`) and reading the data child directly, since there is nothing to prune with.
    /// This covers both legacy zero-length zones and layouts whose aggregates the session cannot
    /// reconstruct.
    fn zoned_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        if self.zone_len == 0 {
            return self
                .slot(0)?
                .vortex_expect("ZonedLayout always has a data child")
                .new_reader(name, segment_source, session, ctx);
        }

        Ok(Arc::new(ZonedReader::try_new(
            self.clone(),
            self.nzones(),
            name,
            segment_source,
            session.clone(),
            ctx.clone(),
        )?))
    }

    pub fn nzones(&self) -> usize {
        usize::try_from(self.children().child_row_count(1))
            .vortex_expect("Invalid number of zones, cannot handle more than usize zones")
    }
}

impl ZonedData {
    pub(crate) fn zone_len(&self) -> usize {
        self.zone_len
    }

    pub(crate) fn aggregate_fns(&self) -> Arc<[AggregateFnRef]> {
        match &self.zone_map_schema {
            ZoneMapSchema::LegacyStats(stats) => stats
                .iter()
                .filter_map(Stat::aggregate_fn)
                .collect::<Vec<_>>()
                .into(),
            ZoneMapSchema::AggregateFns(aggregate_fns) => Arc::clone(aggregate_fns),
        }
    }
}

fn present_aggregates(schema: &ZoneMapSchema) -> Arc<[String]> {
    match schema {
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
    use crate::LayoutBuildContext;
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
            max_bytes: unsafe { NonZeroUsize::new_unchecked(128) },
        });
        let metadata = ZonedMetadata {
            zone_len: 314,
            aggregate_specs: Arc::new([AggregateSpecProto::try_from_aggregate_fn(&aggregate_fn)?]),
        };

        let deserialized = ZonedMetadata::deserialize(&metadata.serialize())?;
        let session = VortexSession::empty().with::<AggregateFnSession>();
        let aggregate_fns = try_aggregate_fns_from_specs(&deserialized.aggregate_specs, &session)?
            .expect("known aggregates resolve");

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

        assert_eq!(layout.zone_len, 0);
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

    #[test]
    fn test_build_unknown_aggregate_disables_pruning_when_allowed() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let read_ctx = ReadContext::new([]);
        let children = OwnedLayoutChildren::layout_children(vec![
            FlatLayout::new(0, dtype.clone(), SegmentId::from(0), read_ctx.clone()).into_layout(),
            FlatLayout::new(0, dtype.clone(), SegmentId::from(1), read_ctx).into_layout(),
        ]);
        let session = vortex_array::array_session();
        session.allow_unknown();
        let build_read_ctx = ReadContext::new([]);
        let build_ctx = LayoutBuildContext {
            session: &session,
            array_read_ctx: &build_read_ctx,
        };

        let metadata = ZonedMetadata {
            zone_len: 8,
            aggregate_specs: Arc::new([AggregateSpecProto::new_unknown("vortex.test.unknown")]),
        };

        let layout = <Zoned as VTable>::build(
            &ZonedLayoutEncoding,
            &dtype,
            0,
            &metadata,
            vec![],
            children.as_ref(),
            &build_ctx,
        )?;

        // An unknown aggregate disables zoned pruning (`zone_len == 0`) while leaving the data
        // child readable, so the index is ignored rather than turned into a hard read error.
        assert_eq!(layout.zone_len, 0);
        Ok(())
    }

    #[test]
    fn test_build_unknown_aggregate_errors_without_allow_unknown() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let read_ctx = ReadContext::new([]);
        let children = OwnedLayoutChildren::layout_children(vec![
            FlatLayout::new(0, dtype.clone(), SegmentId::from(0), read_ctx.clone()).into_layout(),
            FlatLayout::new(0, dtype.clone(), SegmentId::from(1), read_ctx).into_layout(),
        ]);
        let session = vortex_array::array_session();
        let build_read_ctx = ReadContext::new([]);
        let build_ctx = LayoutBuildContext {
            session: &session,
            array_read_ctx: &build_read_ctx,
        };

        let metadata = ZonedMetadata {
            zone_len: 8,
            aggregate_specs: Arc::new([AggregateSpecProto::new_unknown("vortex.test.unknown")]),
        };

        let result = <Zoned as VTable>::build(
            &ZonedLayoutEncoding,
            &dtype,
            0,
            &metadata,
            vec![],
            children.as_ref(),
            &build_ctx,
        );

        assert!(result.is_err());
    }
}
