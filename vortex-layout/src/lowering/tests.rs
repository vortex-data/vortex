// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::stats::Stat;
use vortex_array::stats::as_stat_bitset_bytes;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::ArrayExprExt;
use vortex_expr::gt;
use vortex_expr::lit;
use vortex_expr::root;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_mask::Mask;
use vortex_scan::plan::Filter;
use vortex_scan::plan::*;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_session::registry::ReadContext;

use crate::LayoutBuildContext;
use crate::LayoutDeserialization;
use crate::LayoutPlugin;
use crate::LayoutRef;
use crate::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::dict::DictLayout;
use crate::layouts::flat::FlatLayout;
use crate::layouts::foreign::new_foreign_layout;
use crate::layouts::list::ListLayout;
use crate::layouts::struct_::StructLayout;
use crate::layouts::zoned::LegacyStatsLayoutEncoding;
use crate::layouts::zoned::MAX_IS_TRUNCATED;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::legacy_stats_to_aggregates;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::lowering::LegacyStats;
use crate::lowering::*;
use crate::segments::SegmentId;
use crate::test::SESSION;

fn primitive(ptype: PType, nullability: Nullability) -> DType {
    DType::Primitive(ptype, nullability)
}

fn flat(row_count: u64, dtype: DType, segment: u32) -> LayoutRef {
    FlatLayout::new(
        row_count,
        dtype,
        SegmentId::from(segment),
        ReadContext::new([]),
    )
    .into_layout()
}

fn unsupported(row_count: u64, dtype: DType) -> LayoutRef {
    static ID: CachedId = CachedId::new("vortex.test.unsupported");
    new_foreign_layout(*ID, dtype, row_count, Vec::new(), Vec::new(), Vec::new())
}

fn make_plan(layout: LayoutRef) -> VortexResult<PlanRef> {
    layout.new_plan()
}

fn child_of(plan: &PlanRef, slot: usize) -> VortexResult<PlanRef> {
    plan.slot_required(slot)
}

fn assert_unsupported(error: VortexError) {
    assert!(
        error
            .to_string()
            .contains("No physical plan implementation for layout 'vortex.test.unsupported'"),
        "unexpected error: {error}"
    );
}

#[test]
fn unsupported_layout_has_no_plan() -> VortexResult<()> {
    let layout = unsupported(3, DType::Null);

    assert_unsupported(
        layout
            .new_plan()
            .err()
            .ok_or_else(|| vortex_err!("unsupported layout unexpectedly produced a plan"))?,
    );
    Ok(())
}

#[test]
fn flat_plan_is_a_filtered_segment_scan_without_children() -> VortexResult<()> {
    let plan = make_plan(flat(3, primitive(PType::I32, Nullability::NonNullable), 0))?;

    assert!(plan.is::<Filter>());
    let scan = child_of(&plan, 0)?;
    assert!(scan.is::<SegmentScan>());
    assert_eq!(scan.nslots(), 0);
    assert!(scan.slot(0).is_err());
    Ok(())
}

#[test]
fn chunked_plan_exposes_chunks() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        3,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![flat(2, dtype.clone(), 0), flat(1, dtype, 1)]),
    )
    .into_layout();
    let plan = make_plan(layout)?;

    assert!(plan.is::<Concat>());
    assert_eq!(plan.nslots(), 2);
    assert_eq!(child_of(&plan, 0)?.row_count(), 2);
    assert_eq!(child_of(&plan, 1)?.row_count(), 1);
    Ok(())
}

#[test]
fn chunked_plan_lowers_each_chunk_on_access() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = ChunkedLayout::new(
        2,
        dtype.clone(),
        OwnedLayoutChildren::layout_children(vec![
            flat(1, dtype.clone(), 0),
            unsupported(1, dtype),
        ]),
    )
    .into_layout();

    let plan = make_plan(layout)?;
    assert_eq!(child_of(&plan, 0)?.row_count(), 1);
    assert_unsupported(
        child_of(&plan, 1)
            .err()
            .ok_or_else(|| vortex_err!("unsupported chunk unexpectedly produced a plan"))?,
    );
    Ok(())
}

#[test]
fn struct_plan_lowers_each_field_on_access() -> VortexResult<()> {
    let field_dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = StructLayout::new(
        1,
        DType::Struct(
            StructFields::from_iter([("a", field_dtype.clone()), ("b", field_dtype.clone())]),
            Nullability::NonNullable,
        ),
        vec![flat(1, field_dtype.clone(), 0), unsupported(1, field_dtype)],
    )
    .into_layout();

    let plan = make_plan(layout)?;
    assert_eq!(child_of(&plan, Pack::FIELDS_OFFSET)?.row_count(), 1);
    assert_unsupported(
        child_of(&plan, Pack::FIELDS_OFFSET + 1)
            .err()
            .ok_or_else(|| vortex_err!("unsupported field unexpectedly produced a plan"))?,
    );
    Ok(())
}

#[test]
fn dict_plan_orders_codes_before_values() -> VortexResult<()> {
    let values_dtype = primitive(PType::I32, Nullability::NonNullable);
    let codes_dtype = primitive(PType::U8, Nullability::NonNullable);
    let layout = DictLayout::new(
        flat(2, values_dtype.clone(), 0),
        flat(3, codes_dtype.clone(), 1),
    )
    .into_layout();
    let plan = make_plan(layout)?;

    assert!(plan.is::<Take>());
    assert_eq!(plan.nslots(), 2);
    assert_eq!(child_of(&plan, 0)?.dtype(), &codes_dtype);
    assert_eq!(child_of(&plan, 1)?.dtype(), &values_dtype);
    Ok(())
}

#[test]
fn list_plan_validity_child_is_present_when_nullable() -> VortexResult<()> {
    let element_dtype = primitive(PType::I32, Nullability::NonNullable);
    let offsets_dtype = primitive(PType::U32, Nullability::NonNullable);
    let non_nullable = ListLayout::new(
        DType::List(Arc::new(element_dtype.clone()), Nullability::NonNullable),
        flat(3, element_dtype.clone(), 0),
        flat(3, offsets_dtype.clone(), 1),
        None,
    )
    .into_layout();
    let plan = make_plan(non_nullable)?;

    assert!(plan.is::<ListPack>());
    assert_eq!(plan.nslots(), 3);
    assert_eq!(child_of(&plan, 0)?.dtype(), &element_dtype);
    assert_eq!(child_of(&plan, 1)?.dtype(), &offsets_dtype);
    assert!(plan.slot(2)?.is_none());

    let nullable = ListLayout::new(
        DType::List(Arc::new(element_dtype.clone()), Nullability::Nullable),
        flat(3, element_dtype, 2),
        flat(3, offsets_dtype, 3),
        Some(flat(2, DType::Bool(Nullability::NonNullable), 4)),
    )
    .into_layout();
    let nullable_plan = make_plan(nullable)?;
    assert_eq!(nullable_plan.nslots(), 3);
    assert_eq!(
        child_of(&nullable_plan, 2)?.dtype(),
        &DType::Bool(Nullability::NonNullable)
    );
    Ok(())
}

#[test]
fn struct_plan_validity_child_is_present_when_nullable() -> VortexResult<()> {
    let field_dtype = primitive(PType::I32, Nullability::NonNullable);
    let fields = StructFields::from_iter([("a", field_dtype.clone()), ("b", field_dtype.clone())]);
    let non_nullable = StructLayout::new(
        3,
        DType::Struct(fields.clone(), Nullability::NonNullable),
        vec![
            flat(3, field_dtype.clone(), 0),
            flat(3, field_dtype.clone(), 1),
        ],
    )
    .into_layout();
    let plan = make_plan(non_nullable)?;

    assert!(plan.is::<Pack>());
    assert_eq!(plan.nslots(), 3);
    assert!(plan.slot(Pack::VALIDITY_SLOT)?.is_none());
    assert_eq!(child_of(&plan, Pack::FIELDS_OFFSET)?.dtype(), &field_dtype);
    assert_eq!(
        child_of(&plan, Pack::FIELDS_OFFSET + 1)?.dtype(),
        &field_dtype
    );

    let nullable = StructLayout::new(
        3,
        DType::Struct(fields, Nullability::Nullable),
        vec![
            flat(3, DType::Bool(Nullability::NonNullable), 2),
            flat(3, field_dtype.clone(), 3),
            flat(3, field_dtype, 4),
        ],
    )
    .into_layout();
    let nullable_plan = make_plan(nullable)?;
    assert_eq!(nullable_plan.nslots(), 3);
    assert_eq!(
        child_of(&nullable_plan, Pack::VALIDITY_SLOT)?.dtype(),
        &DType::Bool(Nullability::NonNullable)
    );
    Ok(())
}

#[test]
fn zoned_plan_exposes_data_and_zones() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let (max, zones_dtype) = max_zones_dtype(&dtype)?;
    let zone_len = NonZeroUsize::new(3).ok_or_else(|| vortex_err!("zone length is zero"))?;
    let layout = ZonedLayout::try_new(
        flat(5, dtype, 0),
        flat(2, zones_dtype, 1),
        zone_len,
        vec![max].into(),
    )?
    .into_layout();

    let plan = make_plan(layout)?;
    assert!(plan.is::<Zoned>());
    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.zoned(i32, rows=5)
      data: vortex.plan.filter(i32, rows=5)
        child: vortex.plan.segment_scan(i32, rows=5)
      zones: vortex.plan.filter({vortex.max()=i32?}, rows=2)
        child: vortex.plan.segment_scan({vortex.max()=i32?}, rows=2)
    ");
    Ok(())
}

/// A legacy stats layout over 5 rows of `dtype` in zones of `zone_len` rows, whose zone table of
/// `zones_dtype` holds `stats`.
fn legacy_layout(
    dtype: &DType,
    zone_len: u32,
    stats: &[Stat],
    zones_dtype: DType,
) -> VortexResult<LayoutRef> {
    let children = OwnedLayoutChildren::layout_children(vec![
        flat(5, dtype.clone(), 0),
        flat(2, zones_dtype, 1),
    ]);
    let session = VortexSession::empty();
    let read_ctx = ReadContext::new([]);
    let build_ctx = LayoutBuildContext {
        session: &session,
        array_read_ctx: &read_ctx,
    };
    let mut metadata = zone_len.to_le_bytes().to_vec();
    metadata.extend(as_stat_bitset_bytes(stats));
    LayoutPlugin::deserialize(
        &LegacyStatsLayoutEncoding,
        LayoutDeserialization {
            serialized_id: LayoutPlugin::id(&LegacyStatsLayoutEncoding),
            dtype,
            row_count: 5,
            metadata: &metadata,
            segment_ids: Vec::new(),
            children: children.as_ref(),
        },
        &build_ctx,
    )
}

#[test]
fn legacy_stats_layout_lowers_to_legacy_stats_plan() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let layout = legacy_layout(&dtype, 3, &[Stat::Max], legacy_max_zones_dtype(&dtype))?;

    let plan = make_plan(layout)?;
    assert!(plan.is::<LegacyStats>());
    insta::assert_snapshot!(plan.display_tree(), @"
    root: vortex.plan.legacy_stats(i32, rows=5)
      data: vortex.plan.filter(i32, rows=5)
        child: vortex.plan.segment_scan(i32, rows=5)
      zones: vortex.plan.filter({max=i32?, max_is_truncated=bool}, rows=2)
        child: vortex.plan.segment_scan({max=i32?, max_is_truncated=bool}, rows=2)
    ");
    Ok(())
}

/// Zoned layouts with no zones, or a zone table holding no statistics, lower to their data.
#[test]
fn zoned_layouts_without_statistics_lower_to_their_data() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let empty = DType::Struct(StructFields::empty(), Nullability::NonNullable);
    let zone_len = NonZeroUsize::new(3).ok_or_else(|| vortex_err!("zone length is zero"))?;
    let layouts = [
        legacy_layout(&dtype, 0, &[Stat::Max], legacy_max_zones_dtype(&dtype))?,
        legacy_layout(&dtype, 3, &[], empty.clone())?,
        ZonedLayout::try_new(
            flat(5, dtype.clone(), 0),
            flat(2, empty, 1),
            zone_len,
            Vec::new().into(),
        )?
        .into_layout(),
    ];
    for layout in layouts {
        let plan = make_plan(layout)?;
        assert!(plan.is::<Filter>(), "{}", plan.display_tree());
    }
    Ok(())
}

/// A legacy zone table holding the max of a `dtype` column.
fn legacy_max_zones_dtype(dtype: &DType) -> DType {
    DType::Struct(
        StructFields::from_iter([
            (Stat::Max.name(), dtype.as_nullable()),
            (MAX_IS_TRUNCATED, DType::Bool(Nullability::NonNullable)),
        ]),
        Nullability::NonNullable,
    )
}

/// Legacy stats over a 5-row i32 column in 3-row zones, holding each zone's max.
fn legacy_max_plan(dtype: &DType) -> VortexResult<PlanRef> {
    Ok(LegacyStats::new_plan(
        make_plan(flat(5, dtype.clone(), 0))?,
        make_plan(flat(2, legacy_max_zones_dtype(dtype), 1))?,
        3,
        vec![Stat::Max].into(),
    )
    .into_plan())
}

/// The aggregate zone table of a max over `dtype`, and the max aggregate.
fn max_zones_dtype(dtype: &DType) -> VortexResult<(AggregateFnRef, DType)> {
    let max = Max.bind(NumericalAggregateOpts::skip_nans());
    let max_dtype = max
        .state_dtype(dtype)
        .ok_or_else(|| vortex_err!("max does not support {dtype}"))?;
    let zones_dtype = DType::Struct(
        StructFields::from_iter([(max.to_string(), max_dtype.as_nullable())]),
        Nullability::NonNullable,
    );
    Ok((max, zones_dtype))
}

#[test]
fn legacy_stats_rewrite_to_zoned_over_converted_zones() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let (max, zones_dtype) = max_zones_dtype(&dtype)?;
    let legacy = legacy_max_plan(&dtype)?;

    // Read as data, the legacy stats are dropped along with the zoned plan they convert to.
    assert!(optimize(legacy.clone(), &SESSION)?.is::<Filter>());

    let falsifier = gt(root(), lit(5_i32))
        .bind(&dtype)?
        .falsify(&SESSION)?
        .ok_or_else(|| vortex_err!("filter has no falsifier"))?;
    let pruning = optimize(EvalPlan::try_new(falsifier, legacy)?.into_plan(), &SESSION)?;
    let pruning = pruning
        .as_opt::<Zoned>()
        .ok_or_else(|| vortex_err!("legacy stats predicate did not become a pruning plan"))?;
    assert!(pruning.is_pruning());
    assert_eq!(pruning.zone_len(), 3);
    assert_eq!(
        pruning
            .aggregate_fns()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        vec![max.to_string()]
    );
    let zones = pruning.zones_plan()?;
    assert!(zones.is::<Pack>());
    assert_eq!(zones.dtype(), &zones_dtype);
    Ok(())
}

#[test]
fn converted_legacy_zones_prune() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let table_dtype = legacy_max_zones_dtype(&dtype);
    let (aggregate_fns, conversion) =
        legacy_stats_to_aggregates(&dtype, &table_dtype, &[Stat::Max]);
    let table = StructArray::from_fields(&[
        (
            Stat::Max.name(),
            PrimitiveArray::from_option_iter([Some(3_i32), Some(9)]).into_array(),
        ),
        (
            MAX_IS_TRUNCATED,
            BoolArray::from_iter([false, true]).into_array(),
        ),
    ])?;
    assert_eq!(table.dtype(), &table_dtype);

    let converted = table
        .into_array()
        .apply_bound(&conversion.bind(&table_dtype)?)?
        .execute::<StructArray>(&mut SESSION.create_execution_ctx())?;
    let zone_map = ZoneMap::try_new(dtype.clone(), converted, aggregate_fns, 3, 5)?;
    let falsifier = gt(root(), lit(5_i32))
        .bind(&dtype)?
        .falsify(&SESSION)?
        .ok_or_else(|| vortex_err!("filter has no falsifier"))?;
    // Only the first zone's max proves no row exceeds 5.
    assert_eq!(
        zone_map.prune(&falsifier, &SESSION)?,
        Mask::from_iter([true, false])
    );
    Ok(())
}

/// A zoned plan over legacy stats read as data keeps neither zone table.
#[test]
fn zoned_over_legacy_stats_read_as_data_keeps_only_the_data() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let (max, zones_dtype) = max_zones_dtype(&dtype)?;
    let zones = make_plan(flat(2, zones_dtype, 2))?;
    let legacy = legacy_max_plan(&dtype)?;
    let data = LegacyStats::data_plan(
        legacy
            .as_opt::<LegacyStats>()
            .ok_or_else(|| vortex_err!("legacy plan is not legacy stats"))?,
    )?;
    let plan = ZonedPlan::new(legacy, zones, 3, vec![max].into()).into_plan();

    assert!(PlanRef::ptr_eq(&optimize(plan, &SESSION)?, &data));
    Ok(())
}

/// Legacy stats over a zoned plan are dropped: read as data only the data remains, and a zone
/// pruning plan is evaluated over the aggregate zones, never the legacy table.
#[test]
fn legacy_stats_over_zoned_are_dropped() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let (max, zones_dtype) = max_zones_dtype(&dtype)?;
    let data = make_plan(flat(5, dtype.clone(), 0))?;
    let zones = make_plan(flat(2, zones_dtype, 2))?;
    let zoned = ZonedPlan::new(data.clone(), zones.clone(), 3, vec![max].into()).into_plan();
    let legacy = LegacyStats::new_plan(
        zoned,
        make_plan(flat(2, legacy_max_zones_dtype(&dtype), 1))?,
        3,
        vec![Stat::Max].into(),
    )
    .into_plan();

    assert!(PlanRef::ptr_eq(&optimize(legacy.clone(), &SESSION)?, &data));

    let falsifier = gt(root(), lit(5_i32))
        .bind(&dtype)?
        .falsify(&SESSION)?
        .ok_or_else(|| vortex_err!("filter has no falsifier"))?;
    let pruning = optimize(EvalPlan::try_new(falsifier, legacy)?.into_plan(), &SESSION)?;
    let pruning = pruning
        .as_opt::<Zoned>()
        .ok_or_else(|| vortex_err!("predicate over legacy stats did not become a pruning plan"))?;
    assert!(pruning.is_pruning());
    assert!(PlanRef::ptr_eq(&pruning.zones_plan()?, &zones));
    Ok(())
}

#[test]
fn legacy_stats_with_zero_zone_length_keep_only_data() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let data = make_plan(flat(5, dtype.clone(), 0))?;
    let legacy = LegacyStats::new_plan(
        data.clone(),
        make_plan(flat(2, legacy_max_zones_dtype(&dtype), 1))?,
        0,
        vec![Stat::Max].into(),
    )
    .into_plan();

    assert!(PlanRef::ptr_eq(&optimize(legacy, &SESSION)?, &data));
    Ok(())
}

#[cfg(debug_assertions)]
#[test]
fn legacy_stats_rewrites_validate_output_shape() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let data = make_plan(flat(5, dtype.clone(), 0))?;
    for (declared_dtype, declared_rows, expected_error) in [
        (dtype, 6, "changed row count"),
        (
            primitive(PType::U64, Nullability::NonNullable),
            5,
            "changed dtype",
        ),
    ] {
        let legacy = LegacyStats::from_slots(
            declared_dtype,
            declared_rows,
            [data.clone(), data.clone()].into(),
            0,
            Arc::new([]),
        );
        let error = LegacyStats::reduce(&legacy, &SESSION).unwrap_err();
        assert!(error.to_string().contains(expected_error), "{error}");
    }
    Ok(())
}

#[cfg(debug_assertions)]
#[test]
fn legacy_stats_parent_rewrites_validate_output_shape() -> VortexResult<()> {
    let dtype = primitive(PType::I32, Nullability::NonNullable);
    let data = make_plan(flat(5, dtype.clone(), 0))?;
    let legacy = LegacyStats::from_slots(
        dtype.clone(),
        6,
        [data.clone(), data].into(),
        0,
        Arc::new([]),
    );
    let falsifier = gt(root(), lit(5_i32))
        .bind(&dtype)?
        .falsify(&SESSION)?
        .ok_or_else(|| vortex_err!("filter has no falsifier"))?;
    let parent = EvalPlan::try_new(falsifier, legacy.clone().into_plan())?.into_plan();
    let error = LegacyStats::reduce_parent(&legacy, &parent, 0, &SESSION).unwrap_err();
    assert!(error.to_string().contains("changed row count"), "{error}");
    Ok(())
}
