// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::expr::GetItem;
use vortex_array::expr::Statistic;
use vortex_array::expr::stats::Stat;
use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_error::VortexResult;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStreamRef;
use crate::v2::readers::scalar_fn::ScalarFnReaderExt;

pub struct ZonedReader {
    data: ReaderRef,
    zone_map: ReaderRef,
    zone_len: usize,
    present_stats: Arc<[Stat]>,
}

impl Reader for ZonedReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.data.dtype()
    }

    fn row_count(&self) -> u64 {
        self.data.row_count()
    }

    fn try_reduce_parent(
        &self,
        parent: &ReaderRef,
        _child_idx: usize,
    ) -> VortexResult<Option<ReaderRef>> {
        if let Some(stat) = parent.as_scalar_fn::<Statistic>() {
            if !self.present_stats.contains(stat) {
                return Ok(None);
            }

            // We know the statistic is present; so we return a new reader that pulls the value
            // from the zone map.
            let zoned_statistic = GetItem.new_reader(
                // FIXME(ngates): construct the field name properly
                FieldName::from(stat.name()),
                vec![self.zone_map.clone()],
                self.zone_map.row_count(),
            )?;

            // We now need to explode the zoned_statistic to match the data reader's row count.
            // We do this based on the zone map's zone length.
            let exploded_statistic = Arc::new(ZonedExpansionReader {
                zoned: zoned_statistic,
                zone_len: self.zone_len,
                row_count: self.data.row_count(),
            });

            return Ok(Some(exploded_statistic));
        }

        Ok(None)
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        // By default, a zoned reader is just a pass-through.
        self.data.execute(row_range)
    }
}

/// A reader that expands zoned statistics to match the data rows.
/// This repeats each row of the zone map `zone_len` times.
/// TODO(ngates): we could use a RunEndReader + Slice to do this?
struct ZonedExpansionReader {
    zoned: ReaderRef,
    zone_len: usize,
    row_count: u64,
}

impl Reader for ZonedExpansionReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.zoned.dtype()
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        todo!()
    }
}
