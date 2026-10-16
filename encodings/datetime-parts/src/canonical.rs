// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSource;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_compute::lane_kernels::LaneZip;
use vortex_compute::lane_kernels::Repeat;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::array::DateTimePartsParts;
use crate::timestamp::SECONDS_PER_DAY;

/// Decode a [`DateTimePartsParts`] back into a [`TemporalArray`].
///
/// The three components are combined in a single pass over the rows. Constant seconds or
/// subseconds, which `split_temporal` produces for second-precision data, are folded into the
/// kernel instead of materialized.
pub fn decode_to_temporal(
    parts: DateTimePartsParts,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<TemporalArray> {
    let DType::Extension(ext) = dtype else {
        vortex_panic!(Compute: "expected dtype to be DType::Extension variant")
    };

    let Some(options) = ext.metadata_opt::<Timestamp>() else {
        vortex_panic!(Compute: "must decode TemporalMetadata from extension metadata");
    };

    let divisor = match options.unit {
        TimeUnit::Nanoseconds => 1_000_000_000,
        TimeUnit::Microseconds => 1_000_000,
        TimeUnit::Milliseconds => 1_000,
        TimeUnit::Seconds => 1,
        TimeUnit::Days => vortex_panic!(InvalidArgument: "cannot decode into TimeUnit::D"),
    };

    let days = parts.days.as_::<Primitive>();
    let validity = days.validity()?;
    let len = days.len();

    let seconds = TimePart::try_new(&parts.seconds, ctx)?;
    let subseconds = TimePart::try_new(&parts.subseconds, ctx)?;

    let mut values = BufferMut::<i64>::with_capacity_in(len, ctx.allocator().clone());
    let out = &mut values.spare_capacity_mut()[..len];

    match_each_integer_ptype!(days.ptype(), |D| {
        let days = days.as_slice::<D>();
        match (&seconds, &subseconds) {
            (TimePart::Constant(seconds), TimePart::Constant(subseconds)) => combine_parts(
                days,
                Repeat::new(*seconds, len),
                Repeat::new(*subseconds, len),
                divisor,
                out,
            ),
            (TimePart::Constant(seconds), TimePart::Values(subseconds)) => combine_parts(
                days,
                Repeat::new(*seconds, len),
                subseconds.as_slice::<i32>(),
                divisor,
                out,
            ),
            (TimePart::Values(seconds), TimePart::Constant(subseconds)) => combine_parts(
                days,
                seconds.as_slice::<i32>(),
                Repeat::new(*subseconds, len),
                divisor,
                out,
            ),
            (TimePart::Values(seconds), TimePart::Values(subseconds)) => combine_parts(
                days,
                seconds.as_slice::<i32>(),
                subseconds.as_slice::<i32>(),
                divisor,
                out,
            ),
        }
    });
    // SAFETY: `combine_parts` writes every lane of `out`, which spans exactly `len` items.
    unsafe { values.set_len(len) };

    Ok(TemporalArray::new_timestamp(
        PrimitiveArray::new(values.freeze(), validity).into_array(),
        options.unit,
        options.tz.clone(),
    ))
}

/// A non-nullable integer component of every timestamp.
enum TimePart {
    /// One value shared by every row.
    Constant(i64),
    /// A materialized `i32` column.
    Values(PrimitiveArray),
}

impl TimePart {
    fn try_new(part: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(constant) = part.as_constant() {
            return Ok(Self::Constant(
                constant
                    .as_primitive()
                    .as_::<i64>()
                    .vortex_expect("non-nullable"),
            ));
        }

        // `split_temporal` produces `i32` parts, so this cast is normally a no-op. Other widths
        // are cast up front rather than instantiating the fused kernel for every width
        // combination.
        let values = part
            .cast(DType::Primitive(PType::I32, Nullability::NonNullable))?
            .execute::<PrimitiveArray>(ctx)?;
        Ok(Self::Values(values))
    }
}

/// Writes `days * 86_400 * divisor + seconds * divisor + subseconds` for every lane of `out`.
fn combine_parts<D, S, U>(
    days: &[D],
    seconds: S,
    subseconds: U,
    divisor: i64,
    out: &mut [MaybeUninit<i64>],
) where
    D: Copy + AsPrimitive<i64>,
    S: IndexedSource,
    S::Item: AsPrimitive<i64>,
    U: IndexedSource,
    U::Item: AsPrimitive<i64>,
{
    let day_scale = SECONDS_PER_DAY * divisor;
    LaneZip::new(LaneZip::new(days, seconds), subseconds).map_into(
        out,
        |((day, second), subsecond)| {
            day.as_() * day_scale + second.as_() * divisor + subsecond.as_()
        },
    );
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::DateTimeParts;
    use crate::array::DateTimePartsArraySlotsExt;
    use crate::array::DateTimePartsParts;
    use crate::canonical::decode_to_temporal;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    /// Executes a datetime-parts array and returns its timestamp storage values.
    fn execute_storage(
        array: vortex_array::ArrayRef,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        array
            .execute::<Canonical>(ctx)?
            .into_extension()
            .storage_array()
            .clone()
            .execute::<PrimitiveArray>(ctx)
    }

    #[rstest]
    #[case(Validity::NonNullable)]
    #[case(Validity::AllValid)]
    #[case(Validity::AllInvalid)]
    #[case(Validity::from_iter([true, true, false, false, true, true]))]
    fn test_decode_to_temporal(#[case] validity: Validity) -> VortexResult<()> {
        let milliseconds = PrimitiveArray::new(
            buffer![
                86_400i64, // element with only day component
                -86_400i64,
                86_400i64 + 1000, // element with day + second components
                -86_400i64 - 1000,
                86_400i64 + 1000 + 1, // element with day + second + sub-second components
                -86_400i64 - 1000 - 1
            ],
            validity.clone(),
        );
        let mut ctx = SESSION.create_execution_ctx();
        let date_times = DateTimeParts::try_from_temporal(
            TemporalArray::new_timestamp(
                milliseconds.clone().into_array(),
                TimeUnit::Milliseconds,
                Some("UTC".into()),
            ),
            &mut ctx,
        )?;

        assert!(date_times.as_array().validity()?.mask_eq(
            &validity,
            milliseconds.len(),
            &mut ctx
        )?);

        let dtype = date_times.dtype().clone();
        let parts = DateTimePartsParts {
            days: date_times.days().clone(),
            seconds: date_times.seconds().clone(),
            subseconds: date_times.subseconds().clone(),
        };

        let primitive_values = decode_to_temporal(parts, &dtype, &mut ctx)?
            .temporal_values()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        assert_arrays_eq!(primitive_values, milliseconds, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::seconds(TimeUnit::Seconds)]
    #[case::milliseconds(TimeUnit::Milliseconds)]
    #[case::nanoseconds(TimeUnit::Nanoseconds)]
    fn decode_round_trips_many_rows(#[case] unit: TimeUnit) -> VortexResult<()> {
        let divisor: i64 = match unit {
            TimeUnit::Seconds => 1,
            TimeUnit::Milliseconds => 1_000,
            TimeUnit::Microseconds => 1_000_000,
            TimeUnit::Nanoseconds => 1_000_000_000,
            TimeUnit::Days => unreachable!(),
        };
        // Cover several days, both signs, and lengths that leave a partial chunk.
        let timestamps = PrimitiveArray::from_iter(
            (-1_500i64..1_500).map(|i| i * 7_919 * divisor + (i.rem_euclid(97)) * (divisor / 97)),
        );
        let mut ctx = SESSION.create_execution_ctx();
        let date_times = DateTimeParts::try_from_temporal(
            TemporalArray::new_timestamp(timestamps.clone().into_array(), unit, None),
            &mut ctx,
        )?;

        let decoded = execute_storage(date_times.into_array(), &mut ctx)?;
        assert_arrays_eq!(decoded, timestamps, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::constant_seconds(true, false)]
    #[case::constant_subseconds(false, true)]
    #[case::both_constant(true, true)]
    fn decode_folds_constant_parts(
        #[case] constant_seconds: bool,
        #[case] constant_subseconds: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        const LEN: i32 = 200;
        let len = LEN as usize;
        let days = PrimitiveArray::from_iter((0..LEN).map(|i| i - 100));
        let seconds = if constant_seconds {
            ConstantArray::new(3_600i32, len).into_array()
        } else {
            PrimitiveArray::from_iter((0..LEN).map(|i| i * 400)).into_array()
        };
        let subseconds = if constant_subseconds {
            ConstantArray::new(250i32, len).into_array()
        } else {
            PrimitiveArray::from_iter((0..LEN).map(|i| i * 3)).into_array()
        };

        let expected = PrimitiveArray::from_iter((0..i64::from(LEN)).map(|i| {
            let seconds = if constant_seconds { 3_600 } else { i * 400 };
            let subseconds = if constant_subseconds { 250 } else { i * 3 };
            (i - 100) * 86_400_000 + seconds * 1_000 + subseconds
        }));
        let dtype = TemporalArray::new_timestamp(
            expected.clone().into_array(),
            TimeUnit::Milliseconds,
            None,
        )
        .dtype()
        .clone();

        let decoded = execute_storage(
            DateTimeParts::try_new(dtype, days.into_array(), seconds, subseconds)?.into_array(),
            &mut ctx,
        )?;
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn decode_casts_narrow_parts() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let days = PrimitiveArray::from_iter([0i16, 1, -1]);
        let seconds = PrimitiveArray::from_iter([5u8, 6, 7]);
        let subseconds = PrimitiveArray::from_iter([1i64, 2, 3]);

        let expected =
            PrimitiveArray::from_iter([5_001i64, 86_400_000 + 6_002, -86_400_000 + 7_003]);
        let dtype = TemporalArray::new_timestamp(
            expected.clone().into_array(),
            TimeUnit::Milliseconds,
            None,
        )
        .dtype()
        .clone();

        let decoded = execute_storage(
            DateTimeParts::try_new(
                dtype,
                days.into_array(),
                seconds.into_array(),
                subseconds.into_array(),
            )?
            .into_array(),
            &mut ctx,
        )?;
        assert_arrays_eq!(decoded, expected, &mut ctx);
        Ok(())
    }
}
