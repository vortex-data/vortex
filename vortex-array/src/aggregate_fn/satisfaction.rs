// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Whether a stored aggregate function can satisfy a requested aggregate function.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AggregateFnSatisfaction {
    /// The stored aggregate cannot satisfy the requested aggregate.
    #[default]
    No,
    /// The stored aggregate can satisfy the request as an approximate bound.
    Approximate,
    /// The stored aggregate exactly satisfies the request.
    Exact,
}

impl AggregateFnSatisfaction {
    /// Returns whether the stored aggregate can satisfy the requested aggregate.
    pub fn is_satisfied(self) -> bool {
        !matches!(self, Self::No)
    }

    /// Returns whether the stored aggregate exactly satisfies the requested aggregate.
    pub fn is_exact(self) -> bool {
        matches!(self, Self::Exact)
    }
}
