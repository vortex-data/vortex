// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::DateTimeParts;
use crate::array::DateTimePartsArrayExt;
fn take_datetime_parts(
    array: ArrayView<DateTimeParts>,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // we go ahead and canonicalize here to avoid worst-case canonicalizing 3 separate times
    let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;

    let taken_days = array.days().take(indices.clone().into_array())?;
    let taken_seconds = array.seconds().take(indices.clone().into_array())?;
    let taken_subseconds = array.subseconds().take(indices.clone().into_array())?;

    // Update the dtype if the nullability changed due to nullable indices
    let dtype = if taken_days.dtype().is_nullable() != array.dtype().is_nullable() {
        array
            .dtype()
            .with_nullability(taken_days.dtype().nullability())
    } else {
        array.dtype().clone()
    };

    if !taken_seconds.dtype().is_nullable() && !taken_subseconds.dtype().is_nullable() {
        return Ok(
            DateTimeParts::try_new(dtype, taken_days, taken_seconds, taken_subseconds)?
                .into_array(),
        );
    }

    // DateTimePartsArray requires seconds and subseconds to be non-nullable.
    // If they became nullable due to nullable indices, we need to fill nulls.
    // But first, we need to check that the types are consistent.
    if !taken_days.dtype().is_nullable() {
        vortex_panic!("Mismatched types: days is not nullable, seconds is nullable");
    }
    if !taken_seconds.dtype().is_nullable() {
        vortex_panic!("Mismatched types: seconds is not nullable, days is nullable");
    }
    if !taken_subseconds.dtype().is_nullable() {
        vortex_panic!("Mismatched types: subseconds is not nullable, days & seconds are nullable");
    }
    if !indices.dtype().is_nullable() {
        vortex_panic!("Mismatched types: indices are not nullable, days & seconds are nullable");
    }

    let seconds_fill = array
        .seconds()
        .statistics()
        .get(Stat::Min)
        .map(|s| s.into_inner())
        .unwrap_or_else(|| Scalar::primitive(0i64, Nullability::NonNullable))
        .cast(array.seconds().dtype())?;
    let taken_seconds = taken_seconds.fill_null(seconds_fill)?;

    let subseconds_fill = array
        .subseconds()
        .statistics()
        .get(Stat::Min)
        .map(|s| s.into_inner())
        .unwrap_or_else(|| Scalar::primitive(0i64, Nullability::NonNullable))
        .cast(array.subseconds().dtype())?;
    let taken_subseconds = taken_subseconds.fill_null(subseconds_fill)?;

    Ok(DateTimeParts::try_new(dtype, taken_days, taken_seconds, taken_subseconds)?.into_array())
}

impl TakeExecute for DateTimeParts {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        take_datetime_parts(array, indices, ctx).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_buffer::buffer;

    use crate::DateTimeParts;
    use crate::DateTimePartsArray;

    #[rstest]
    #[case(DateTimeParts::try_from_temporal(TemporalArray::new_timestamp(
        buffer![
            0i64,
            86_400_000,  // 1 day in ms
            172_800_000, // 2 days in ms
            259_200_000, // 3 days in ms
            345_600_000, // 4 days in ms
        ].into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".into())
    ), &mut LEGACY_SESSION.create_execution_ctx()).unwrap())]
    #[case(DateTimeParts::try_from_temporal(TemporalArray::new_timestamp(
        PrimitiveArray::from_option_iter([
            Some(0i64),
            None,
            Some(172_800_000), // 2 days in ms
            Some(259_200_000), // 3 days in ms
            None,
        ]).into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".into())
    ), &mut LEGACY_SESSION.create_execution_ctx()).unwrap())]
    #[case(DateTimeParts::try_from_temporal(TemporalArray::new_timestamp(
        buffer![86_400_000i64].into_array(),
        TimeUnit::Milliseconds,
        Some("UTC".into())
    ), &mut LEGACY_SESSION.create_execution_ctx()).unwrap())]
    fn test_take_datetime_parts_conformance(#[case] array: DateTimePartsArray) {
        test_take_conformance(&array.into_array());
    }
}
