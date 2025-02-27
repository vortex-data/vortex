use vortex_scalar::Scalar;

use crate::Array;
use crate::compute::{IsConstantOpts, is_constant_opts, scalar_at};
use crate::stats::StatsSetRef;

/// Extension functions for arrays that provide statistics.
pub trait ArrayStatisticsExt {
    /// Make a best effort attempt to try and figure out if the array is constant, without canonicalizing it.
    fn is_constant(&self) -> bool;

    /// If [`ArrayStatistics::is_constant`] is true, return the actual constant value as a [`Scalar`].
    fn as_constant(&self) -> Option<Scalar>;
}

impl<A: Array + 'static> ArrayStatisticsExt for A {
    fn is_constant(&self) -> bool {
        static SKIP_CANONICALIZE: IsConstantOpts = IsConstantOpts {
            canonicalize: false,
        };
        is_constant_opts(self, &SKIP_CANONICALIZE)
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
