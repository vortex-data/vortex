// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::DateTimeParts;
use crate::DateTimePartsSlots;
use crate::timestamp;
use crate::timestamp::TimestampParts;

impl OperationsVTable<DateTimeParts> for DateTimeParts {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, DateTimeParts>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let DType::Extension(ext) = array.dtype().clone() else {
            vortex_panic!(
                "DateTimePartsArray must have extension dtype, found {}",
                array.dtype()
            );
        };

        let Some(options) = ext.metadata_opt::<Timestamp>() else {
            vortex_panic!("must decode TemporalMetadata from extension metadata");
        };

        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(DType::Extension(ext)));
        }

        let days = part_at(state, DateTimePartsSlots::DAYS, "days", index, ctx)?;
        let seconds = part_at(state, DateTimePartsSlots::SECONDS, "seconds", index, ctx)?;
        let subseconds = part_at(
            state,
            DateTimePartsSlots::SUBSECONDS,
            "subseconds",
            index,
            ctx,
        )?;

        let ts = timestamp::combine(
            TimestampParts {
                days,
                seconds,
                subseconds,
            },
            options.unit,
        )?;

        Ok(Scalar::extension::<Timestamp>(
            options.clone(),
            Scalar::primitive(ts, ext.storage_dtype().nullability()),
        ))
    }

    fn scalar_at(
        array: ArrayView<'_, DateTimeParts>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

/// Reads one timestamp part out of `slot`, through the probe so a repeated read keeps the
/// child's preparation.
fn part_at<T: IntegerPType>(
    state: &mut ProbeState<'_, DateTimeParts>,
    slot: usize,
    name: &'static str,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<T> {
    state
        .slot(slot)?
        .ok_or_else(|| vortex_err!("DateTimeParts {name} slot is missing"))?
        .execute_scalar(index, ctx)?
        .as_primitive()
        .as_opt::<T>()
        .flatten()
        .ok_or_else(|| {
            vortex_err!(
                "DateTimeParts {name} at index {index} is null or does not fit {}",
                T::PTYPE
            )
        })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;

    #[rstest]
    #[case::days_overflow(i64::MAX, 0, "overflow")]
    #[case::seconds_out_of_range(0, i64::from(i32::MAX) + 1, "seconds at index 0")]
    fn test_scalar_invalid_parts(
        #[case] days: i64,
        #[case] seconds: i64,
        #[case] message: &str,
    ) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let array = DateTimeParts::try_new(
            DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullability::NonNullable).erased()),
            buffer![days].into_array(),
            ConstantArray::new(seconds, 1).into_array(),
            ConstantArray::new(0i32, 1).into_array(),
        )?;
        let error = array.execute_scalar(0, &mut ctx).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");

        Ok(())
    }
}
