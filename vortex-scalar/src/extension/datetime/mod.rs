// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension scalar definitions for concrete datetime types.

use vortex_dtype::datetime::AnyTemporal;
use vortex_dtype::datetime::Date;
use vortex_dtype::datetime::Time;
use vortex_dtype::datetime::TimeUnit;
use vortex_dtype::datetime::Timestamp;

mod date;
mod time;
mod timestamp;

pub use date::*;
pub use time::*;
pub use timestamp::*;

use crate::ExtScalar;
use crate::extension::Matcher;

/// A matched temporal storage value, produced by [`AnyTemporal`]'s [`Matcher`] impl.
pub enum TemporalValue<'a> {
    /// A [`Timestamp`] value.
    Timestamp(TimestampValue<'a>),
    /// A [`Date`] value.
    Date(DateValue),
    /// A [`Time`] value.
    Time(TimeValue),
}

impl Matcher for AnyTemporal {
    /// Extract the matched temporal storage value as an i64.
    type Match<'a> = Option<TemporalValue<'a>>;

    fn try_match<'a>(item: &'a ExtScalar) -> Option<Self::Match<'a>> {
        if let Some(value) = item.as_value_opt::<Timestamp>() {
            return Some(value.map(TemporalValue::Timestamp));
        }

        if let Some(value) = item.as_value_opt::<Date>() {
            return Some(value.map(TemporalValue::Date));
        }

        if let Some(value) = item.as_value_opt::<Time>() {
            return Some(value.map(TemporalValue::Time));
        }

        None
    }
}

/// Convenience constructor for building a [`jiff::Span`] from a [`TimeUnit`] and length.
trait SpanExt {
    /// Create a span of the given `length` in the specified [`TimeUnit`].
    fn from_unit_length(length: i64, time_unit: TimeUnit) -> Self;
}

impl SpanExt for jiff::Span {
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
