// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Write;
use std::fmt::{self};

use crate::ArrayRef;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;

/// Display wrapper for array statistics in compact format.
///
/// Produces output like ` [nulls=3, min=5, max=100]` (with leading space).
pub(crate) struct StatsDisplay<'a>(pub(crate) &'a ArrayRef);

impl fmt::Display for StatsDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.0.statistics();
        let mut first = true;

        let mut sep = |f: &mut fmt::Formatter<'_>| -> fmt::Result {
            if first {
                first = false;
                f.write_str(" [")
            } else {
                f.write_str(", ")
            }
        };

        // Null count or validity fallback
        if let Some(nc) = stats.get(Stat::NullCount) {
            if let Ok(n) = usize::try_from(&nc.clone().into_inner()) {
                sep(f)?;
                write!(f, "nulls={}", n)?;
            } else {
                sep(f)?;
                write!(f, "nulls={}", nc)?;
            }
        } else if self.0.dtype().is_nullable() {
            match self.0.all_valid() {
                Ok(true) => {
                    sep(f)?;
                    f.write_str("all_valid")?;
                }
                Ok(false) => {
                    if self.0.all_invalid().unwrap_or(false) {
                        sep(f)?;
                        f.write_str("all_invalid")?;
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to check validity: {e}");
                    sep(f)?;
                    f.write_str("validity_failed")?;
                }
            }
        }

        // NaN count (only if > 0)
        if let Some(nan) = stats.get(Stat::NaNCount)
            && let Ok(n) = usize::try_from(&nan.into_inner())
            && n > 0
        {
            sep(f)?;
            write!(f, "nan={}", n)?;
        }

        // Min/Max
        if let Some(min) = stats.get(Stat::Min) {
            sep(f)?;
            write!(f, "min={}", min)?;
        }
        if let Some(max) = stats.get(Stat::Max) {
            sep(f)?;
            write!(f, "max={}", max)?;
        }

        // Sum
        if let Some(sum) = stats.get(Stat::Sum) {
            sep(f)?;
            write!(f, "sum={}", sum)?;
        }

        // Boolean flags (compact)
        if let Some(c) = stats.get(Stat::IsConstant)
            && bool::try_from(&c.into_inner()).unwrap_or(false)
        {
            sep(f)?;
            f.write_str("const")?;
        }
        if let Some(s) = stats.get(Stat::IsStrictSorted) {
            if bool::try_from(&s.into_inner()).unwrap_or(false) {
                sep(f)?;
                f.write_str("strict")?;
            }
        } else if let Some(s) = stats.get(Stat::IsSorted)
            && bool::try_from(&s.into_inner()).unwrap_or(false)
        {
            sep(f)?;
            f.write_str("sorted")?;
        }

        // Close bracket if we wrote anything
        if !first {
            f.write_char(']')?;
        }

        Ok(())
    }
}

/// Extractor that adds stats annotations (e.g. `[nulls=3, min=5]`) to the header line.
pub struct StatsExtractor;

impl TreeExtractor for StatsExtractor {
    fn write_header(
        &self,
        array: &ArrayRef,
        _ctx: &TreeContext,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}", StatsDisplay(array))
    }
}
