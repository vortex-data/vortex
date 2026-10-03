// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Write;
use std::fmt::{self};

use crate::ArrayRef;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::is_constant::IS_CONSTANT;
use crate::aggregate_fn::fns::is_sorted::IS_SORTED;
use crate::aggregate_fn::fns::is_sorted::IS_STRICT_SORTED;
use crate::aggregate_fn::fns::max::MAX_SKIP_NANS;
use crate::aggregate_fn::fns::min::MIN_SKIP_NANS;
use crate::aggregate_fn::fns::nan_count::NAN_COUNT;
use crate::aggregate_fn::fns::null_count::NULL_COUNT;
use crate::aggregate_fn::fns::sum::SUM_SKIP_NANS;
use crate::display::extractor::TreeContext;
use crate::display::extractor::TreeExtractor;
use crate::validity::Validity;

/// Display wrapper for array statistics in compact format.
///
/// Produces output like ` [nulls=3, min=5, max=100]` (with leading space).
pub(crate) struct StatsDisplay<'a>(pub(crate) &'a ArrayRef);

impl fmt::Display for StatsDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.0.aggregations();
        // Historical display omits null results, including known overflowing sums.
        let get_non_null = |aggregate: &AggregateFnRef| {
            stats
                .get_result(aggregate)
                .into_inner()
                .filter(|s| !s.is_null())
        };
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
        if let Some(nc) = get_non_null(&NULL_COUNT) {
            if let Ok(n) = usize::try_from(&nc) {
                sep(f)?;
                write!(f, "nulls={}", n)?;
            } else {
                sep(f)?;
                write!(f, "nulls={}", nc)?;
            }
        } else if self.0.dtype().is_nullable() {
            match self.0.validity() {
                Ok(Validity::NonNullable | Validity::AllValid) => {
                    sep(f)?;
                    f.write_str("all_valid")?;
                }
                Ok(Validity::AllInvalid) => {
                    sep(f)?;
                    f.write_str("all_invalid")?;
                }
                Ok(Validity::Array(_)) => {
                    // Avoid computing validity-array stats as a side effect of display.
                }
                Err(e) => {
                    tracing::warn!("Failed to check validity: {e}");
                    sep(f)?;
                    f.write_str("validity_failed")?;
                }
            }
        }

        // NaN count (only if > 0)
        if let Some(nan) = get_non_null(&NAN_COUNT)
            && let Ok(n) = usize::try_from(&nan)
            && n > 0
        {
            sep(f)?;
            write!(f, "nan={}", n)?;
        }

        // Min/Max
        if let Some(min) = get_non_null(&MIN_SKIP_NANS) {
            sep(f)?;
            write!(f, "min={}", min)?;
        }
        if let Some(max) = get_non_null(&MAX_SKIP_NANS) {
            sep(f)?;
            write!(f, "max={}", max)?;
        }

        // Sum
        if let Some(sum) = get_non_null(&SUM_SKIP_NANS) {
            sep(f)?;
            write!(f, "sum={}", sum)?;
        }

        // Boolean flags (compact)
        if let Some(c) = get_non_null(&IS_CONSTANT)
            && bool::try_from(&c).unwrap_or(false)
        {
            sep(f)?;
            f.write_str("const")?;
        }
        if let Some(s) = get_non_null(&IS_STRICT_SORTED) {
            if bool::try_from(&s).unwrap_or(false) {
                sep(f)?;
                f.write_str("strict")?;
            }
        } else if let Some(s) = get_non_null(&IS_SORTED)
            && bool::try_from(&s).unwrap_or(false)
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

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::StatsDisplay;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::is_constant::is_constant;
    use crate::aggregate_fn::fns::is_sorted::is_sorted;
    use crate::aggregate_fn::fns::min::MIN_SKIP_NANS;
    use crate::aggregate_fn::fns::min_max::min_max;
    use crate::aggregate_fn::fns::sum::sum;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;

    #[test]
    fn finalized_results_keep_compact_display() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = buffer![7i32, 7].into_array();
        array.valid_count(&mut ctx)?;
        min_max(&array, &mut ctx, NumericalAggregateOpts::skip_nans())?;
        sum(&array, &mut ctx)?;
        is_constant(&array, &mut ctx)?;
        is_sorted(&array, &mut ctx)?;

        let before = array.aggregations().snapshot_results();
        assert_eq!(
            format!("{}", StatsDisplay(&array)),
            " [nulls=0, min=7i32, max=7i32, sum=14i64, const, sorted]"
        );
        let after = array.aggregations().snapshot_results();
        assert_eq!(
            before.iter().collect::<Vec<_>>(),
            after.iter().collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn exact_null_extrema_are_omitted() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = PrimitiveArray::from_option_iter([None::<i32>, None]).into_array();
        assert!(
            array
                .aggregations()
                .compute_result(&MIN_SKIP_NANS, &mut ctx)?
                .is_null()
        );

        assert_eq!(format!("{}", StatsDisplay(&array)), " [nulls=2]");
        Ok(())
    }

    #[test]
    fn exact_null_sum_is_omitted() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = buffer![i64::MAX, 1].into_array();
        assert!(sum(&array, &mut ctx)?.is_null());

        assert_eq!(format!("{}", StatsDisplay(&array)), "");
        Ok(())
    }
}
