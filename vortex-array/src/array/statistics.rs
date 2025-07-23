// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;

use crate::Array;
use crate::compute::{Cost, IsConstantOpts, is_constant_opts};

impl dyn Array {
    /// Returns true if for all pairs of valid indices `i` and `j` in the array,
    /// `self.scalar_at(i) == self.scalar_at(j)`.
    ///
    /// # Notes
    ///
    /// - For arrays with less than two valid indices, the array is considered constant.
    /// - This is a convenience method that uses the `Specialized` cost optimization.
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

    /// Returns true if for all pairs of valid indices `i` and `j` in the array,
    /// `self.scalar_at(i) == self.scalar_at(j)`, with specified cost optimization.
    ///
    /// # Notes
    ///
    /// - For arrays with less than two valid indices, the array is considered constant.
    pub fn is_constant_opts(&self, cost: Cost) -> bool {
        let opts = IsConstantOpts { cost };
        is_constant_opts(self, &opts)
            .inspect_err(|e| log::warn!("Failed to compute IsConstant: {e}"))
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    /// Returns the constant value if the array is constant and non-empty, otherwise None.
    ///
    /// Note that for empty arrays, the result will also be None.
    pub fn as_constant(&self) -> Option<Scalar> {
        self.is_constant().then(|| self.scalar_at(0).ok()).flatten()
    }
}
