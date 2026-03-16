// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::TemporalArray;
use vortex::array::dtype::FieldNames;
use vortex::array::extension::datetime::TimeUnit;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
use vortex::buffer::Buffer;
use vortex::encodings::datetime_parts::DateTimeParts;
use vortex::encodings::datetime_parts::DateTimePartsArray;
use vortex::encodings::datetime_parts::split_temporal;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct DateTimePartsFixture;

fn encode_temporal(temporal: TemporalArray) -> VortexResult<ArrayRef> {
    let dtype = temporal.dtype().clone();
    let parts = split_temporal(temporal)?;
    Ok(
        DateTimePartsArray::try_new(dtype, parts.days, parts.seconds, parts.subseconds)?
            .into_array(),
    )
}

impl FlatLayoutFixture for DateTimePartsFixture {
    fn name(&self) -> &str {
        "datetimeparts.vortex"
    }

    fn description(&self) -> &str {
        "Timestamp arrays for DateTimeParts encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![DateTimeParts::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let base_us: i64 = 1_704_067_200_000_000;
        let ts_us: Vec<i64> = (0..N as i64).map(|i| base_us + i * 3_600_000_000).collect();
        let ts_us_arr = TemporalArray::new_timestamp(
            PrimitiveArray::new(Buffer::from(ts_us), Validity::NonNullable).into_array(),
            TimeUnit::Microseconds,
            None,
        );

        let base_ns: i64 = 1_704_067_200_000_000_000;
        let ts_ns: Vec<i64> = (0..N as i64).map(|i| base_ns + i * 1_000_000_000).collect();
        let ts_ns_arr = TemporalArray::new_timestamp(
            PrimitiveArray::new(Buffer::from(ts_ns), Validity::NonNullable).into_array(),
            TimeUnit::Nanoseconds,
            None,
        );

        let base_ms: i64 = 1_704_067_200_000;
        let ts_ms: Vec<i64> = (0..N as i64).map(|i| base_ms + i * 1000).collect();
        let ts_ms_arr = TemporalArray::new_timestamp(
            PrimitiveArray::new(Buffer::from(ts_ms), Validity::NonNullable).into_array(),
            TimeUnit::Milliseconds,
            None,
        );

        let ts_us_nullable = PrimitiveArray::from_option_iter(
            (0..N as i64).map(|i| (i % 10 != 0).then(|| base_us + i * 60_000_000)),
        );
        let ts_us_nullable_arr =
            TemporalArray::new_timestamp(ts_us_nullable.into_array(), TimeUnit::Microseconds, None);

        let base_s: i64 = 1_704_067_200;
        let ts_s: Vec<i64> = (0..N as i64).map(|i| base_s + i * 86400).collect();
        let ts_s_arr = TemporalArray::new_timestamp(
            PrimitiveArray::new(Buffer::from(ts_s), Validity::NonNullable).into_array(),
            TimeUnit::Seconds,
            None,
        );
        let ts_ms_tz: Vec<i64> = (0..N as i64).map(|i| base_ms + i * 60_000).collect();
        let ts_ms_tz_arr = TemporalArray::new_timestamp(
            PrimitiveArray::new(Buffer::from(ts_ms_tz), Validity::NonNullable).into_array(),
            TimeUnit::Milliseconds,
            Some("UTC".into()),
        );
        let ts_pre_1970: Vec<i64> = (0..N as i64).map(|i| -86_400_000 + i * 1000).collect();
        let ts_pre_1970_arr = TemporalArray::new_timestamp(
            PrimitiveArray::new(Buffer::from(ts_pre_1970), Validity::NonNullable).into_array(),
            TimeUnit::Milliseconds,
            None,
        );
        let ts_day_boundary: Vec<i64> = (0..N as i64)
            .map(|i| base_ms + (i / 4) * 86_400_000 + [0, 999, 1000, 86_399_999][(i % 4) as usize])
            .collect();
        let ts_day_boundary_arr = TemporalArray::new_timestamp(
            PrimitiveArray::new(Buffer::from(ts_day_boundary), Validity::NonNullable).into_array(),
            TimeUnit::Milliseconds,
            None,
        );
        let ts_ns_subsecond: Vec<i64> = (0..N as i64)
            .map(|i| {
                base_ns + (i / 4) * 1_000_000_000 + [0, 1, 999_999, 999_999_999][(i % 4) as usize]
            })
            .collect();
        let ts_ns_subsecond_arr = TemporalArray::new_timestamp(
            PrimitiveArray::new(Buffer::from(ts_ns_subsecond), Validity::NonNullable).into_array(),
            TimeUnit::Nanoseconds,
            None,
        );
        let ts_head_tail_null = PrimitiveArray::from_option_iter((0..N as i64).map(|i| {
            if i < 8 || i >= N as i64 - 8 {
                None
            } else {
                Some(base_ms + i * 1000)
            }
        }));
        let ts_head_tail_null_arr = TemporalArray::new_timestamp(
            ts_head_tail_null.into_array(),
            TimeUnit::Milliseconds,
            None,
        );
        let arr = StructArray::try_new(
            FieldNames::from([
                "ts_us",
                "ts_ns",
                "ts_ms",
                "ts_us_nullable",
                "ts_s",
                "ts_ms_tz",
                "ts_pre_1970",
                "ts_day_boundary",
                "ts_ns_subsecond",
                "ts_head_tail_null",
            ]),
            vec![
                encode_temporal(ts_us_arr)?,
                encode_temporal(ts_ns_arr)?,
                encode_temporal(ts_ms_arr)?,
                encode_temporal(ts_us_nullable_arr)?,
                encode_temporal(ts_s_arr)?,
                encode_temporal(ts_ms_tz_arr)?,
                encode_temporal(ts_pre_1970_arr)?,
                encode_temporal(ts_day_boundary_arr)?,
                encode_temporal(ts_ns_subsecond_arr)?,
                encode_temporal(ts_head_tail_null_arr)?,
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
