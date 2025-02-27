use vortex_scalar::Scalar;

use crate::Array;
use crate::compute::{IsConstantOpts, is_constant_opts, scalar_at};
use crate::stats::StatsSetRef;

/// Extension functions for arrays that provide statistics.
pub trait ArrayStatistics {
    /// Make a best effort attempt to try and figure out if the array is constant, without canonicalizing it.
    fn is_constant(&self) -> bool;

    /// If [`ArrayStatistics::is_constant`] is true, return the actual constant value as a [`Scalar`].
    fn as_constant(&self) -> Option<Scalar>;
}

impl<A: Array + 'static> ArrayStatistics for A {
    fn is_constant(&self) -> bool {
        let opts = IsConstantOpts {
            canonicalize: false,
        };
        is_constant_opts(self, &opts)
            .inspect_err(|e| log::warn!("Failed to compute IsConstant: {e}"))
            .ok()
            .unwrap_or_default()
    }

    fn as_constant(&self) -> Option<Scalar> {
        self.is_constant()
            .then(|| scalar_at(self, 0).ok())
            .flatten()
    }
}

pub trait ArrayStatisticsImpl {
    fn _stats_ref(&self) -> StatsSetRef<'_>;
}
