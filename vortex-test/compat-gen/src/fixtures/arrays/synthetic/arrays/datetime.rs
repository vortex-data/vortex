// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::dtype::FieldNames;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct DateTimeFixture;

impl FlatLayoutFixture for DateTimeFixture {
    fn name(&self) -> &str {
        "datetime.vortex"
    }

    fn description(&self) -> &str {
        "Temporal arrays covering timestamps, dates, and times with various units"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Extension::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        // Timestamps in seconds (i64): 2024-01-01T00:00:00Z, 2024-06-15T12:30:00Z, 2024-12-31T23:59:59Z
        let ts_seconds = TemporalArray::new_timestamp(
            PrimitiveArray::new(
                buffer![1704067200i64, 1718451000, 1735689599],
                Validity::NonNullable,
            )
            .into_array(),
            TimeUnit::Seconds,
            None,
        );

        // Timestamps in milliseconds with timezone (i64)
        let ts_millis_tz = TemporalArray::new_timestamp(
            PrimitiveArray::new(
                buffer![1704067200000i64, 1718451000000, 1735689599000],
                Validity::NonNullable,
            )
            .into_array(),
            TimeUnit::Milliseconds,
            Some(Arc::from("UTC")),
        );

        // Nullable timestamps in microseconds (i64)
        let ts_nullable = TemporalArray::new_timestamp(
            PrimitiveArray::from_option_iter([
                Some(1704067200000000i64),
                None,
                Some(1735689599000000i64),
            ])
            .into_array(),
            TimeUnit::Microseconds,
            None,
        );

        // Date in days since epoch (i32): 2024-01-01, 2024-06-15, 2024-12-31
        let date_days = TemporalArray::new_date(
            PrimitiveArray::new(buffer![19723i32, 19889, 19723 + 365], Validity::NonNullable)
                .into_array(),
            TimeUnit::Days,
        );

        // Time in seconds since midnight (i32): 00:00:00, 12:30:00, 23:59:59
        let time_secs = TemporalArray::new_time(
            PrimitiveArray::new(buffer![0i32, 45000, 86399], Validity::NonNullable).into_array(),
            TimeUnit::Seconds,
        );

        // Timestamps in nanoseconds (i64)
        let ts_nanos = TemporalArray::new_timestamp(
            PrimitiveArray::new(
                buffer![
                    1704067200000000000i64,
                    1718451000000000000,
                    1735689599000000000
                ],
                Validity::NonNullable,
            )
            .into_array(),
            TimeUnit::Nanoseconds,
            None,
        );

        // Timestamps with non-UTC timezone
        let ts_eastern = TemporalArray::new_timestamp(
            PrimitiveArray::new(
                buffer![1704067200i64, 1718451000, 1735689599],
                Validity::NonNullable,
            )
            .into_array(),
            TimeUnit::Seconds,
            Some(Arc::from("America/New_York")),
        );

        let arr = StructArray::try_new(
            FieldNames::from([
                "ts_seconds",
                "ts_millis_tz",
                "ts_nullable_micros",
                "date_days",
                "time_seconds",
                "ts_nanos",
                "ts_eastern",
            ]),
            vec![
                ts_seconds.into_array(),
                ts_millis_tz.into_array(),
                ts_nullable.into_array(),
                date_days.into_array(),
                time_secs.into_array(),
                ts_nanos.into_array(),
                ts_eastern.into_array(),
            ],
            3,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
