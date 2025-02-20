use std::sync::Arc;

use vortex_error::VortexResult;

use crate::stats::{Stat, StatsSet};
use crate::{Array, ArrayImpl};

pub trait ArrayStatistics {
    fn is_constant(&self) -> bool {
        todo!()
    }
}

impl ArrayStatistics for Arc<dyn Array> {}

pub trait ArrayStatisticsImpl {
    fn compute_statistic(&self, stat: Stat) -> VortexResult<StatsSet>;
}

impl<A: ArrayImpl> ArrayStatistics for A {}
