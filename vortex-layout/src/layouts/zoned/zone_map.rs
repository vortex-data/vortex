// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native compatibility helpers for zone statistics tables.

use std::ops::Deref;
use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Stat;
use vortex_error::VortexResult;

use crate::layouts::zoned::schema::legacy_stats_table_dtype;

/// A zone map with native legacy schema compatibility.
#[derive(Clone)]
pub struct ZoneMap {
    inner: vortex_scan::execution::zone_map::ZoneMap,
}

impl Deref for ZoneMap {
    type Target = vortex_scan::execution::zone_map::ZoneMap;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ZoneMap {
    /// Validates an aggregate statistics table before constructing its zone map.
    pub fn try_new(
        column_dtype: DType,
        array: StructArray,
        aggregate_fns: Arc<[AggregateFnRef]>,
        zone_len: u64,
        row_count: u64,
    ) -> VortexResult<Self> {
        Ok(Self {
            inner: vortex_scan::execution::zone_map::ZoneMap::try_new(
                column_dtype,
                array,
                aggregate_fns,
                zone_len,
                row_count,
            )?,
        })
    }

    pub(super) unsafe fn new_unchecked(
        column_dtype: DType,
        array: StructArray,
        aggregate_fns: Arc<[AggregateFnRef]>,
        zone_len: u64,
        row_count: u64,
    ) -> Self {
        Self {
            // SAFETY: The caller validates the native statistics schema.
            inner: unsafe {
                vortex_scan::execution::zone_map::ZoneMap::new_unchecked(
                    column_dtype,
                    array,
                    aggregate_fns,
                    zone_len,
                    row_count,
                )
            },
        }
    }

    /// Returns the [`DType`] of the statistics table given a set of statistics and column [`DType`].
    ///
    /// This remains as a compatibility wrapper around the zoned schema helper.
    #[deprecated(note = "use aggregate-function zoned stats instead")]
    pub fn dtype_for_stats_table(column_dtype: &DType, present_stats: &[Stat]) -> DType {
        legacy_stats_table_dtype(column_dtype, present_stats)
    }
}
