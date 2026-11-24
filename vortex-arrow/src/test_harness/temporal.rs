// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporal transport against independently mapped Arrow types.
//!
//! Source values stay inside native date and time ranges because logical array equality decodes
//! temporal scalars. Numeric storage generation remains free to exercise the full primitive range.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Int32Array;
    use arrow_array::Int64Array;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::TimeUnit as ArrowTimeUnit;
    use hegel::TestCase;
    use hegel::generators as gs;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::PType;
    use vortex_array::extension::datetime::Date;
    use vortex_array::extension::datetime::Time;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::test_harness::StorageCase;
    use crate::test_harness::check_generated;
    use crate::test_harness::generate::draw_validity;

    #[derive(Clone, Copy, Debug)]
    enum TemporalKind {
        Date,
        Time,
        Timestamp,
    }

    #[track_caller]
    fn check_temporal_round_trip(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        let mut ctx = session.create_execution_ctx();

        let len = tc.draw(gs::integers::<usize>().max_value(32));
        let nullability = tc.draw(gs::booleans()).into();
        let date_unit = tc.draw(gs::sampled_from(&[TimeUnit::Days, TimeUnit::Milliseconds]));
        let time_unit = tc.draw(gs::sampled_from(&[
            TimeUnit::Seconds,
            TimeUnit::Milliseconds,
            TimeUnit::Microseconds,
            TimeUnit::Nanoseconds,
        ]));
        let timezone = tc.draw(gs::sampled_from(&[
            None,
            Some("UTC"),
            Some("America/New_York"),
            Some("+05:30"),
        ]));
        let kind = tc.draw(gs::sampled_from(&[
            TemporalKind::Date,
            TemporalKind::Time,
            TemporalKind::Timestamp,
        ]));
        let dtype = match kind {
            TemporalKind::Date => Date::try_new(date_unit, nullability)?.erased(),
            TemporalKind::Time => Time::try_new(time_unit, nullability)?.erased(),
            TemporalKind::Timestamp => {
                Timestamp::try_new_with_tz(time_unit, timezone.map(Into::into), nullability)?
                    .erased()
            }
        };
        let (unit, units_per_second) = match time_unit {
            TimeUnit::Seconds => (ArrowTimeUnit::Second, 1i64),
            TimeUnit::Milliseconds => (ArrowTimeUnit::Millisecond, 1_000),
            TimeUnit::Microseconds => (ArrowTimeUnit::Microsecond, 1_000_000),
            TimeUnit::Nanoseconds => (ArrowTimeUnit::Nanosecond, 1_000_000_000),
            TimeUnit::Days => unreachable!("time units exclude days"),
        };
        let arrow_dtype = match kind {
            TemporalKind::Date => {
                if date_unit == TimeUnit::Days {
                    DataType::Date32
                } else {
                    DataType::Date64
                }
            }
            TemporalKind::Time => {
                if matches!(time_unit, TimeUnit::Seconds | TimeUnit::Milliseconds) {
                    DataType::Time32(unit)
                } else {
                    DataType::Time64(unit)
                }
            }
            TemporalKind::Timestamp => DataType::Timestamp(unit, timezone.map(Into::into)),
        };

        // Dates stay within the native calendar's supported years around the Unix epoch.
        // Fifty years of timestamps fit i64 even in nanoseconds; time values cover one day.
        let (min, max) = match kind {
            TemporalKind::Date => {
                let day = if date_unit == TimeUnit::Days {
                    1
                } else {
                    86_400_000i64
                };
                (-365 * 7_000 * day, 365 * 7_000 * day)
            }
            TemporalKind::Time => (0, 86_400 * units_per_second - 1),
            TemporalKind::Timestamp => (
                -365 * 50 * 86_400 * units_per_second,
                365 * 50 * 86_400 * units_per_second,
            ),
        };
        let validity = draw_validity(tc, dtype.nullability(), len);
        let mask = validity.execute_mask(len, &mut ctx)?;
        let nulls = dtype
            .is_nullable()
            .then(|| NullBuffer::from((0..len).map(|i| mask.value(i)).collect::<Vec<_>>()));
        let storage = if dtype.storage_dtype().as_ptype() == PType::I32 {
            let min = i32::try_from(min).vortex_expect("date-day and time32 bounds fit i32");
            let max = i32::try_from(max).vortex_expect("date-day and time32 bounds fit i32");
            let values = (0..len)
                .map(|_| tc.draw(gs::integers::<i32>().min_value(min).max_value(max)))
                .collect::<Vec<_>>();
            StorageCase {
                array: PrimitiveArray::new(Buffer::from(values.clone()), validity).into_array(),
                arrow: Arc::new(Int32Array::new(values.into(), nulls)),
            }
        } else {
            let values = (0..len)
                .map(|_| tc.draw(gs::integers::<i64>().min_value(min).max_value(max)))
                .collect::<Vec<_>>();
            StorageCase {
                array: PrimitiveArray::new(Buffer::from(values.clone()), validity).into_array(),
                arrow: Arc::new(Int64Array::new(values.into(), nulls)),
            }
        };
        let expected = arrow_array::make_array(
            storage
                .arrow
                .to_data()
                .into_builder()
                .data_type(arrow_dtype.clone())
                .build()?,
        );
        let field = Field::new("value", arrow_dtype, dtype.is_nullable());
        let array = ExtensionArray::try_new(dtype, storage.array)?.into_array();
        check_generated(tc, array, &field, expected, &mut ctx)
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn temporal_arrow_transport(tc: TestCase) {
        check_temporal_round_trip(&tc)
            .vortex_expect("temporal Arrow transport must preserve dtype and stored values");
    }
}
