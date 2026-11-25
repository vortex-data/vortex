// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rounding modes for a temporal count in a coarser unit.
//!
//! `cast` performs only the temporal conversions that are exact for every valid value. A coarser
//! unit can drop part of a count, so the caller selects a [`TemporalRounding`] through
//! `vortex_functions::expr::change_time_unit` instead.

use std::fmt;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;

/// How to handle fractional counts in a coarser temporal unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TemporalRounding {
    /// Require the value to be exactly representable in the target unit.
    Exact,
    /// Discard the remainder toward zero.
    TowardZero,
    /// Round toward negative infinity.
    Floor,
    /// Round to the nearest count, resolving ties away from zero.
    NearestTiesAwayFromZero,
}

impl TemporalRounding {
    /// Divide a signed count by the positive factor between two temporal units.
    pub fn divide(self, value: i64, divisor: i64) -> VortexResult<i64> {
        let quotient = value / divisor;
        let remainder = value % divisor;

        Ok(match self {
            Self::Exact => {
                vortex_ensure_eq!(
                    remainder,
                    0,
                    "Expected an exact temporal conversion, got {value}"
                );
                quotient
            }
            Self::TowardZero => quotient,
            Self::Floor => value.div_euclid(divisor),
            Self::NearestTiesAwayFromZero if remainder.abs() * 2 >= divisor => {
                quotient + remainder.signum()
            }
            Self::NearestTiesAwayFromZero => quotient,
        })
    }
}

impl fmt::Display for TemporalRounding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Exact => "exact",
            Self::TowardZero => "toward_zero",
            Self::Floor => "floor",
            Self::NearestTiesAwayFromZero => "nearest_ties_away_from_zero",
        })
    }
}
