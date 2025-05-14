use vortex_scalar::Scalar;

use crate::Array;
use crate::compute::{Cost, IsConstantOpts, is_constant_opts};

impl dyn Array {
    pub fn is_constant(&self) -> bool {
        let opts = IsConstantOpts {
            cost: Cost::Specialized,
        };
        is_constant_opts(self, &opts)
            .inspect_err(|e| log::warn!("Failed to compute IsConstant: {e}"))
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    pub fn is_constant_opts(&self, cost: Cost) -> bool {
        let opts = IsConstantOpts { cost };
        is_constant_opts(self, &opts)
            .inspect_err(|e| log::warn!("Failed to compute IsConstant: {e}"))
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    pub fn as_constant(&self) -> Option<Scalar> {
        self.is_constant().then(|| self.scalar_at(0).ok()).flatten()
    }
}
