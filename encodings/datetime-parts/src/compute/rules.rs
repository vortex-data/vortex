// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Filter;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::scalar_fn::AnyScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DateTimeParts;
use crate::array::DateTimePartsArraySlotsExt;
use crate::timestamp;
pub(crate) const PARENT_RULES: ParentRuleSet<DateTimeParts> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&DTPFilterPushDownRule),
    ParentRuleSet::lift(&DTPComparisonPushDownRule),
    ParentRuleSet::lift(&CastReduceAdaptor(DateTimeParts)),
    ParentRuleSet::lift(&FilterReduceAdaptor(DateTimeParts)),
    ParentRuleSet::lift(&MaskReduceAdaptor(DateTimeParts)),
    ParentRuleSet::lift(&SliceReduceAdaptor(DateTimeParts)),
]);

/// Push the filter into the days column of a date time parts, we could extend this to other fields
/// but its less clear if that is beneficial.
#[derive(Debug)]
struct DTPFilterPushDownRule;

impl ArrayParentReduceRule<DateTimeParts> for DTPFilterPushDownRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, DateTimeParts>,
        parent: ArrayView<'_, Filter>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        debug_assert_eq!(child_idx, 0);

        if !child.seconds().is::<Constant>() || !child.subseconds().is::<Constant>() {
            return Ok(None);
        }

        DateTimeParts::try_new(
            child.dtype().clone(),
            child.days().clone().filter(parent.filter_mask().clone())?,
            ConstantArray::new(
                child.seconds().as_constant().vortex_expect("constant"),
                parent.filter_mask().true_count(),
            )
            .into_array(),
            ConstantArray::new(
                child.subseconds().as_constant().vortex_expect("constant"),
                parent.filter_mask().true_count(),
            )
            .into_array(),
        )
        .map(|x| Some(x.into_array()))
    }
}

/// Push down comparison operators (Binary and Between) to the days column when both seconds
/// and subseconds are constant zero on both sides of the comparison.
///
/// When a DateTimeParts array has constant zero for seconds and subseconds, and is being
/// compared against a constant timestamp that also has zero seconds and subseconds,
/// we can push the comparison down to just compare the days.
///
/// For example: `dtp <= 2013-07-31` where dtp has seconds=0 and subseconds=0,
/// and the RHS timestamp is exactly at midnight (no time component),
/// becomes: `dtp.days <= 15917` (the day number for 2013-07-31).
#[derive(Debug)]
struct DTPComparisonPushDownRule;

impl ArrayParentReduceRule<DateTimeParts> for DTPComparisonPushDownRule {
    type Parent = AnyScalarFn;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, DateTimeParts>,
        parent: ArrayView<'_, ScalarFn>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only handle comparison operations (Binary comparisons or Between)
        if parent
            .scalar_fn()
            .as_opt::<Binary>()
            .is_none_or(|c| !c.is_comparison())
            && !parent.scalar_fn().is::<Between>()
        {
            return Ok(None);
        }

        // Check that DTP's seconds and subseconds are constant zero
        if !is_constant_zero(child.seconds()) || !is_constant_zero(child.subseconds()) {
            return Ok(None);
        }

        let days = child.days();

        // Build new children: replace DTP with days, replace constant timestamps with days constants
        let mut new_children = Vec::with_capacity(parent.nchildren());
        for (idx, c) in parent.iter_children().enumerate() {
            if idx == child_idx {
                // This is the DTP child - replace with days
                new_children.push(days.clone());
            } else {
                // Must be a constant timestamp at midnight
                let Some(days_value) = try_extract_days_constant(c) else {
                    return Ok(None);
                };
                let len = days.len();
                let target_dtype = days.dtype();
                let constant = ConstantArray::new(days_value, len).into_array();
                new_children.push(constant.cast(target_dtype.clone())?);
            }
        }

        let result = ScalarFnArray::try_new(parent.scalar_fn().clone(), new_children)?
            .into_array()
            .optimize()?;

        Ok(Some(result))
    }
}

/// Try to extract the days value from a constant timestamp.
/// Returns None if the constant is not a timestamp or has non-zero seconds/subseconds.
fn try_extract_days_constant(array: &ArrayRef) -> Option<i64> {
    let constant = array.as_constant()?;

    // Extract the timestamp value
    let timestamp = constant
        .as_extension()
        .to_storage_scalar()
        .as_primitive()
        .as_::<i64>()?;

    // Get the time unit from the dtype
    let DType::Extension(ext_dtype) = constant.dtype() else {
        return None;
    };

    let options = ext_dtype.metadata::<Timestamp>();
    let ts_parts = timestamp::split(timestamp, options.unit).ok()?;

    // Only allow pushdown if seconds and subseconds are zero
    if ts_parts.seconds != 0 || ts_parts.subseconds != 0 {
        return None;
    }

    Some(ts_parts.days)
}

/// Check if an array is a constant with value zero.
fn is_constant_zero(array: &ArrayRef) -> bool {
    array
        .as_opt::<Constant>()
        .is_some_and(|c| c.scalar().is_zero() == Some(true))
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::sum::sum;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::TimestampOptions;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;

    use super::*;
    use crate::DateTimeParts;
    use crate::DateTimePartsArray;

    const SECONDS_PER_DAY: i64 = 86400;

    /// Count the true values in a boolean array using the provided execution context.
    fn true_count(array: &ArrayRef, ctx: &mut ExecutionCtx) -> usize {
        sum(array, ctx)
            .unwrap()
            .as_primitive()
            .as_::<usize>()
            .unwrap()
    }

    /// Create a DTP array with the given day values (all at midnight).
    fn dtp_at_midnight(days: &[i64], time_unit: TimeUnit) -> DateTimePartsArray {
        let multiplier = match time_unit {
            TimeUnit::Seconds => 1,
            TimeUnit::Milliseconds => 1_000,
            TimeUnit::Microseconds => 1_000_000,
            TimeUnit::Nanoseconds => 1_000_000_000,
            TimeUnit::Days => panic!("Days not supported"),
        };
        let timestamps: Vec<i64> = days
            .iter()
            .map(|d| d * SECONDS_PER_DAY * multiplier)
            .collect();
        let buffer: Buffer<i64> = timestamps.into();
        let temporal = TemporalArray::new_timestamp(
            PrimitiveArray::new(buffer, Validity::NonNullable).into_array(),
            time_unit,
            None,
        );
        DateTimeParts::try_from_temporal(temporal, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("TemporalArray must produce valid DateTimeParts")
    }

    /// Create a constant timestamp scalar at midnight for the given day.
    fn midnight_constant(day: i64, time_unit: TimeUnit, len: usize) -> ArrayRef {
        let multiplier = match time_unit {
            TimeUnit::Seconds => 1,
            TimeUnit::Milliseconds => 1_000,
            TimeUnit::Microseconds => 1_000_000,
            TimeUnit::Nanoseconds => 1_000_000_000,
            TimeUnit::Days => panic!("Days not supported"),
        };
        let timestamp = day * SECONDS_PER_DAY * multiplier;
        let scalar = Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: time_unit,
                tz: None,
            },
            timestamp.into(),
        );
        ConstantArray::new(scalar, len).into_array()
    }

    /// Create a constant timestamp scalar with non-midnight time.
    fn non_midnight_constant(day: i64, seconds: i64, time_unit: TimeUnit, len: usize) -> ArrayRef {
        let multiplier = match time_unit {
            TimeUnit::Seconds => 1,
            TimeUnit::Milliseconds => 1_000,
            TimeUnit::Microseconds => 1_000_000,
            TimeUnit::Nanoseconds => 1_000_000_000,
            TimeUnit::Days => panic!("Days not supported"),
        };
        let timestamp = (day * SECONDS_PER_DAY + seconds) * multiplier;
        let scalar = Scalar::extension::<Timestamp>(
            TimestampOptions {
                unit: time_unit,
                tz: None,
            },
            timestamp.into(),
        );
        ConstantArray::new(scalar, len).into_array()
    }

    #[test]
    fn test_binary_comparison_pushdown() {
        // DTP with days [0, 1, 2] at midnight
        let dtp = dtp_at_midnight(&[0, 1, 2], TimeUnit::Seconds);
        let len = dtp.len();

        // Compare: dtp <= day 1 (midnight)
        let constant = midnight_constant(1, TimeUnit::Seconds, len);
        let comparison = Binary
            .try_new_array(len, Operator::Lte, [dtp.into_array(), constant])
            .unwrap();

        // Optimize should push down to days
        let optimized = comparison.optimize().unwrap();

        // The result should be a ScalarFn over primitive days, not over DTP
        assert!(
            !optimized.is::<DateTimeParts>(),
            "Expected pushdown to remove DTP from expression"
        );

        // Verify correctness: days [0, 1, 2] <= 1 should give [true, true, false]
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(true_count(&optimized, &mut ctx), 2);
    }

    #[test]
    fn test_between_pushdown() {
        // DTP with days [0, 1, 2, 3, 4] at midnight
        let dtp = dtp_at_midnight(&[0, 1, 2, 3, 4], TimeUnit::Seconds);
        let len = dtp.len();

        // Between: 1 <= dtp <= 3
        let lower = midnight_constant(1, TimeUnit::Seconds, len);
        let upper = midnight_constant(3, TimeUnit::Seconds, len);

        let between = Between
            .try_new_array(
                len,
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::NonStrict,
                },
                [dtp.into_array(), lower, upper],
            )
            .unwrap();

        // Optimize should push down to days
        let optimized = between.optimize().unwrap();

        // Verify correctness: days [0, 1, 2, 3, 4] between 1 and 3 should give [false, true, true, true, false]
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(true_count(&optimized, &mut ctx), 3);
    }

    #[test]
    fn test_no_pushdown_non_midnight_constant() {
        // DTP with days [0, 1, 2] at midnight
        let dtp = dtp_at_midnight(&[0, 1, 2], TimeUnit::Seconds);
        let len = dtp.len();

        // Compare against non-midnight constant (day 1 at noon)
        let constant = non_midnight_constant(1, 43200, TimeUnit::Seconds, len);
        let comparison = Binary
            .try_new_array(len, Operator::Lte, [dtp.into_array(), constant])
            .unwrap();

        // Optimize should NOT push down (constant has non-zero seconds)
        let optimized = comparison.optimize().unwrap();

        // The DTP should still be in the expression tree
        // (optimization doesn't apply, so we keep the original structure)
        // Just verify it still computes correctly
        // days [0, 1, 2] at midnight <= day 1 at noon: [true, true, false]
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(true_count(&optimized, &mut ctx), 2);
    }

    #[test]
    fn test_no_pushdown_non_zero_dtp_seconds() {
        // Create a DTP with non-zero seconds (not at midnight)
        let timestamps: Buffer<i64> = vec![
            3600,                       // day 0 + 1 hour
            SECONDS_PER_DAY + 3600,     // day 1 + 1 hour
            2 * SECONDS_PER_DAY + 3600, // day 2 + 1 hour
        ]
        .into();
        let temporal = TemporalArray::new_timestamp(
            PrimitiveArray::new(timestamps, Validity::NonNullable).into_array(),
            TimeUnit::Seconds,
            None,
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtp = DateTimeParts::try_from_temporal(temporal, &mut ctx).unwrap();
        let len = dtp.len();

        // Compare against midnight constant
        let constant = midnight_constant(1, TimeUnit::Seconds, len);
        let comparison = Binary
            .try_new_array(len, Operator::Lte, [dtp.into_array(), constant])
            .unwrap();

        // Should still compute correctly (just not optimized via pushdown)
        let optimized = comparison.optimize().unwrap();
        // timestamps at 1am on days [0, 1, 2] <= day 1 midnight: [true, false, false]
        assert_eq!(true_count(&optimized, &mut ctx), 1);
    }
}
