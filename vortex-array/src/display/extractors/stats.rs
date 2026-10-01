// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Write;
use std::fmt::{self};

use crate::ArrayRef;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;
use crate::validity::Validity;

/// Display wrapper for array statistics in compact format.
///
/// Displays each finalized result with its bound function and precision.
pub(crate) struct StatsDisplay<'a>(pub(crate) &'a ArrayRef);

impl fmt::Display for StatsDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let results = self.0.aggregations().snapshot_results();
        let mut first = true;
        let mut sep = |f: &mut fmt::Formatter<'_>| -> fmt::Result {
            if first {
                first = false;
                f.write_str(" [")
            } else {
                f.write_str(", ")
            }
        };

        for (aggregate, value) in results.iter() {
            sep(f)?;
            write!(f, "{aggregate}={value}")?;
        }

        let null_count = AggregateFn::new(NullCount, EmptyOptions).erased();
        if results.get_result(&null_count).is_absent() && self.0.dtype().is_nullable() {
            match self.0.validity() {
                Ok(Validity::NonNullable | Validity::AllValid) => {
                    sep(f)?;
                    f.write_str("all_valid")?;
                }
                Ok(Validity::AllInvalid) => {
                    sep(f)?;
                    f.write_str("all_invalid")?;
                }
                Ok(Validity::Array(_)) => {}
                Err(error) => {
                    tracing::warn!("Failed to check validity: {error}");
                    sep(f)?;
                    f.write_str("validity_failed")?;
                }
            }
        }

        if !first {
            f.write_char(']')?;
        }

        Ok(())
    }
}

/// Extractor that adds finalized aggregate results to the header line.
pub struct StatsExtractor;

impl TreeExtractor<ArrayRef, TreeContext> for StatsExtractor {
    fn write_header(
        &self,
        array: &ArrayRef,
        _ctx: &TreeContext,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}", StatsDisplay(array))
    }
}
