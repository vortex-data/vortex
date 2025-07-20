// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;

use crate::Array;
use crate::compute::{Cost, IsConstantOpts, is_constant_opts};

impl dyn Array {
    /// Returns true if the array contains only constant values.
    ///
    /// This is a convenience method that uses specialized cost optimization.
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

    /// Returns true if the array contains only constant values, with specified cost optimization.
    pub fn is_constant_opts(&self, cost: Cost) -> bool {
        let opts = IsConstantOpts { cost };
        is_constant_opts(self, &opts)
            .inspect_err(|e| log::warn!("Failed to compute IsConstant: {e}"))
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    /// Returns the constant value if the array is constant, otherwise None.
    pub fn as_constant(&self) -> Option<Scalar> {
        self.is_constant().then(|| self.scalar_at(0).ok()).flatten()
    }
}
