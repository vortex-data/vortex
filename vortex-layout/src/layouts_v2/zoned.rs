// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::DeserializeMetadata;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::LayoutChildType;
use crate::LayoutId;
use crate::layout_v2::Layout;
use crate::layout_v2::LayoutDeserializeArgs;
use crate::layout_v2::VTable;
use crate::layouts::zoned::LegacyStatsMetadata;
use crate::layouts::zoned::ZoneMapSchema;
use crate::layouts::zoned::ZonedMetadata;
use crate::layouts::zoned::aggregate_fns_from_specs;
use crate::layouts::zoned::aggregate_stats_table_dtype;
use crate::layouts::zoned::legacy_stats_table_dtype;
use crate::scan::plan::ScanPlanRef;
use crate::scan::plan::request::ScanRequest;
use crate::scan::v2::layouts::zoned as scan_zoned;

/// V2 zoned layout vtable.
#[derive(Clone, Debug)]
pub struct Zoned;

/// V2 legacy stats layout vtable.
#[derive(Clone, Debug)]
pub struct LegacyStats;

/// V2 zoned layout data.
#[derive(Clone, Debug)]
pub struct ZonedData {
    pub(crate) zone_len: usize,
    pub(crate) zone_map_schema: ZoneMapSchema,
    pub(crate) aggregate_fns: Arc<[AggregateFnRef]>,
}

impl ZonedData {
    /// Returns the configured zone length.
    pub fn zone_len(&self) -> usize {
        self.zone_len
    }

    /// Returns the aggregate functions stored in the zone table.
    pub fn aggregate_fns(&self) -> &Arc<[AggregateFnRef]> {
        &self.aggregate_fns
    }

    /// Returns the zone-map schema used by the zone table.
    pub(crate) fn zone_map_schema(&self) -> &ZoneMapSchema {
        &self.zone_map_schema
    }

    fn stats_table_dtype(&self, dtype: &DType) -> DType {
        match &self.zone_map_schema {
            ZoneMapSchema::LegacyStats(stats) => legacy_stats_table_dtype(dtype, stats),
            ZoneMapSchema::AggregateFns(aggregate_fns) => {
                aggregate_stats_table_dtype(dtype, aggregate_fns)
            }
        }
    }
}

impl VTable for Zoned {
    type LayoutData = ZonedData;

    fn id(&self) -> LayoutId {
        LayoutId::new("vortex.zoned")
    }

    fn deserialize(&self, args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        let metadata = ZonedMetadata::deserialize(args.metadata)?;
        let aggregate_fns = aggregate_fns_from_specs(&metadata.aggregate_specs, args.session)?;
        Ok(ZonedData {
            zone_len: metadata.zone_len as usize,
            zone_map_schema: ZoneMapSchema::AggregateFns(Arc::clone(&aggregate_fns)),
            aggregate_fns,
        })
    }

    fn child_dtype(layout: Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            0 => Ok(layout.dtype().clone()),
            1 => Ok(layout.data().stats_table_dtype(layout.dtype())),
            _ => vortex_bail!("Zoned child index out of bounds: {idx}"),
        }
    }

    fn child_type(_layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType> {
        match idx {
            0 => Ok(LayoutChildType::Transparent("data".into())),
            1 => Ok(LayoutChildType::Auxiliary("zones".into())),
            _ => vortex_bail!("Zoned child index out of bounds: {idx}"),
        }
    }

    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef> {
        scan_zoned::new_scan_plan(layout, req, session)
    }
}

impl VTable for LegacyStats {
    type LayoutData = ZonedData;

    fn id(&self) -> LayoutId {
        LayoutId::new("vortex.stats")
    }

    fn deserialize(&self, args: &LayoutDeserializeArgs<'_>) -> VortexResult<Self::LayoutData> {
        let metadata = LegacyStatsMetadata::deserialize(args.metadata)?;
        let aggregate_fns = match &metadata.zone_map_schema {
            ZoneMapSchema::LegacyStats(stats) => stats
                .iter()
                .filter_map(|stat| stat.aggregate_fn())
                .collect::<Vec<_>>()
                .into(),
            ZoneMapSchema::AggregateFns(aggregate_fns) => Arc::clone(aggregate_fns),
        };
        Ok(ZonedData {
            zone_len: metadata.zone_len as usize,
            zone_map_schema: metadata.zone_map_schema,
            aggregate_fns,
        })
    }

    fn child_dtype(layout: Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            0 => Ok(layout.dtype().clone()),
            1 => Ok(layout.data().stats_table_dtype(layout.dtype())),
            _ => vortex_bail!("Legacy stats child index out of bounds: {idx}"),
        }
    }

    fn child_type(_layout: Layout<Self>, idx: usize) -> VortexResult<LayoutChildType> {
        match idx {
            0 => Ok(LayoutChildType::Transparent("data".into())),
            1 => Ok(LayoutChildType::Auxiliary("zones".into())),
            _ => vortex_bail!("Legacy stats child index out of bounds: {idx}"),
        }
    }

    fn new_scan_plan(
        layout: Layout<Self>,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> VortexResult<ScanPlanRef> {
        scan_zoned::new_scan_plan(layout, req, session)
    }
}
