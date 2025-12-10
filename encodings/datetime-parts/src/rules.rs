// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::optimizer::rules::ArrayReduceRule;
use vortex_array::optimizer::rules::Exact;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_dtype::datetime::TemporalMetadata;
use vortex_dtype::datetime::TimeUnit;
use vortex_dtype::DType;
use vortex_dtype::PType;
use vortex_error::vortex_panic;
use vortex_error::VortexResult;

use crate::DateTimePartsArray;
use crate::DateTimePartsVTable;

/// Expand a date-time-parts array into an expression that evaluates to the timestamp.
#[derive(Debug)]
pub(crate) struct DateTimePartsExpandRule;

impl ArrayReduceRule<Exact<DateTimePartsVTable>> for DateTimePartsExpandRule {
    fn matcher(&self) -> Exact<DateTimePartsVTable> {
        Exact::from(&DateTimePartsVTable)
    }

    fn reduce(&self, array: &DateTimePartsArray) -> VortexResult<Option<ArrayRef>> {
        let DType::Extension(ext) = array.dtype().clone() else {
            vortex_panic!(ComputeError: "expected dtype to be DType::Extension variant")
        };

        let Ok(temporal_metadata) = TemporalMetadata::try_from(ext.as_ref()) else {
            vortex_panic!(ComputeError: "must decode TemporalMetadata from extension metadata");
        };

        let divisor: i64 = match temporal_metadata.time_unit() {
            TimeUnit::Nanoseconds => 1_000_000_000,
            TimeUnit::Microseconds => 1_000_000,
            TimeUnit::Milliseconds => 1_000,
            TimeUnit::Seconds => 1,
            TimeUnit::Days => vortex_panic!(InvalidArgument: "cannot decode into TimeUnit::D"),
        };

        // Up-cast days to i64 for computation.
        let days = array
            .days()
            .cast(DType::Primitive(PType::I64, array.dtype().nullability()))?;

        // Multiply days by the number of seconds in a day and the unit divisor.
        let days = days.mul(ConstantArray::new(divisor * 86_400, array.len()).into_array())?;

        // Multiply the seconds by the unit divisor.
        let seconds = array
            .seconds()
            .cast(DType::Primitive(PType::I64, array.dtype().nullability()))?
            .mul(ConstantArray::new(divisor, array.len()).into_array())?;

        // The subseconds are already in the correct unit, just cast to i64.
        let subseconds = array
            .subseconds()
            .cast(DType::Primitive(PType::I64, array.dtype().nullability()))?;

        // Sum the three components together.
        Ok(Some(days.add(seconds)?.add(subseconds)?))
    }
}
