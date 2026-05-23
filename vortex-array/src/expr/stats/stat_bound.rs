// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::bound::IntersectionResult;

/// `StatType` define the bound of a given statistic. (e.g. `Max` is an upper bound),
/// this is used to extract the bound from a `Precision` value, (e.g. `p::bound<Max>()`).
pub trait StatType<T> {
    type Bound: StatBound<T>;

    const STAT: Stat;
}

/// `StatBound` defines the operations that can be performed on a bound.
/// The main bounds are Upper (e.g. max) and Lower (e.g. min).
pub trait StatBound<T>: Sized {
    /// Creates a new bound from a Precision statistic.
    fn lift(value: Precision<T>) -> Self;

    /// Converts `Self` back to `Precision<T>`, inverse of `lift`.
    fn into_value(self) -> Precision<T>;

    /// Finds the smallest bound that covers both bounds.
    /// A.k.a. the `meet` of the bound.
    fn union(&self, other: &Self) -> Option<Self>;

    /// Refines the bounds to the most precise estimate we can make for that bound.
    /// If the bounds are disjoint, then the result is `None`.
    /// e.g. `Precision::Inexact(5)` and `Precision::Exact(6)` would result in `Precision::Inexact(5)`.
    /// A.k.a. the `join` of the bound.
    fn intersection(&self, other: &Self) -> Option<IntersectionResult<Self>>;

    /// Returns the exact value from the bound if that value is exact, otherwise `None`.
    fn to_exact(&self) -> Option<&T>;
}
