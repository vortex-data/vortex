// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::datetime::TimeUnit;

// pub mod date;
// pub mod time;
// pub mod timestamp;

trait SpanExt {
    fn get_unit_length(&self, time_unit: TimeUnit) -> i64;
    fn from_unit_length(length: i64, time_unit: TimeUnit) -> Self;
}

impl SpanExt for jiff::Span {
    fn get_unit_length(&self, time_unit: TimeUnit) -> i64 {
        match time_unit {
            TimeUnit::Nanoseconds => self.get_nanoseconds(),
            TimeUnit::Microseconds => self.get_microseconds(),
            TimeUnit::Milliseconds => self.get_milliseconds(),
            TimeUnit::Seconds => self.get_seconds(),
            TimeUnit::Days => self.get_days() as _,
        }
    }

    fn from_unit_length(length: i64, time_unit: TimeUnit) -> Self {
        match time_unit {
            TimeUnit::Nanoseconds => jiff::Span::new().nanoseconds(length),
            TimeUnit::Microseconds => jiff::Span::new().microseconds(length),
            TimeUnit::Milliseconds => jiff::Span::new().milliseconds(length),
            TimeUnit::Seconds => jiff::Span::new().seconds(length),
            TimeUnit::Days => jiff::Span::new().days(length),
        }
    }
}
