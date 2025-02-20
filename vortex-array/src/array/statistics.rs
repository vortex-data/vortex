use std::sync::Arc;

use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::compute::scalar_at;
use crate::stats::{Precision, Stat, StatsSet};
use crate::{Array, ArrayImpl};

pub trait ArrayStatistics {
    fn is_constant(&self) -> bool;
    fn as_constant(&self) -> Option<Scalar>;
}

impl<A: Array> ArrayStatistics for A {
    fn is_constant(&self) -> bool {
        if let Some(Precision::Exact(is_constant)) = self
            .vtable()
            .compute_statistics(self, Stat::IsConstant)
            .ok()
            .and_then(|stats| stats.get_as::<bool>(Stat::IsConstant))
        {
            is_constant
        } else {
            false
        }
    }

    fn as_constant(&self) -> Option<Scalar> {
        self.is_constant()
            .then(|| scalar_at(self, 0).ok())
            .flatten()
    }
}
