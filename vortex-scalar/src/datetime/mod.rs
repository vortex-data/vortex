// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension scalar definitions for datetime types.

use vortex_dtype::datetime::AnyTemporal;
use vortex_dtype::datetime::Date;
use vortex_dtype::datetime::Time;
use vortex_dtype::datetime::TimeUnit;
use vortex_dtype::datetime::Timestamp;

use crate::ExtScalarRef;
use crate::extension::Matcher;

mod date;
mod time;
mod timestamp;

pub use timestamp::*;

/// A matched temporal scalar value.
pub enum TemporalValue<'a> {
    /// A Time value.
    Time(&'a jiff::civil::Time),
    /// A Date value.
    Date(&'a jiff::civil::Date),
    /// A Timestamp value.
    Timestamp(&'a TimestampValue),
}

impl Matcher for AnyTemporal {
    type Match<'a> = TemporalValue<'a>;

    fn try_match<'a>(item: &'a ExtScalarRef) -> Option<Self::Match<'a>> {
        if let Some(v) = item.value_opt::<Time>() {
            return Some(TemporalValue::Time(v));
        }
        if let Some(v) = item.value_opt::<Date>() {
            return Some(TemporalValue::Date(v));
        }
        if let Some(v) = item.value_opt::<Timestamp>() {
            return Some(TemporalValue::Timestamp(v));
        }
        None
    }
}

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
