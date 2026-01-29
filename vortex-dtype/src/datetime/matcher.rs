// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;

use vortex_error::VortexResult;

use crate::datetime::Date;
use crate::datetime::Time;
use crate::datetime::Timestamp;
use crate::datetime::TimestampOptions;
use crate::extension::ExtDTypeRef;
use crate::extension::ExtDTypeVTable;
use crate::extension::Matcher;

/// Matcher for temporal extension data types.
pub struct AnyTemporal;

impl Matcher for AnyTemporal {
    type Match<'a> = TemporalOptions<'a>;

    fn try_match<'a>(item: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if let Some(opts) = item.try_options::<Timestamp>() {
            return Some(TemporalOptions::Timestamp(opts));
        }
        if let Some(opts) = item.try_options::<Date>() {
            return Some(TemporalOptions::Date(opts));
        }
        if let Some(opts) = item.try_options::<Time>() {
            return Some(TemporalOptions::Time(opts));
        }
        None
    }
}

/// Options for temporal extension data types.
#[derive(Debug, PartialEq, Eq)]
pub enum TemporalOptions<'a> {
    /// Options for Timestamp dtypes
    Timestamp(&'a <Timestamp as ExtDTypeVTable>::Options),
    /// Options for Date dtypes
    Date(&'a <Date as ExtDTypeVTable>::Options),
    /// Options for Time dtypes
    Time(&'a <Time as ExtDTypeVTable>::Options),
}

// TODO(ngates): remove this logic in favor of having an ExtScalarVTable in vortex-scalar.
//  Currently this is used largely to implement scalar display hacks.
impl TemporalOptions<'_> {
    /// Get the time unit of the temporal dtype.
    pub fn time_unit(&self) -> crate::datetime::TimeUnit {
        match self {
            TemporalOptions::Time(unit) => **unit,
            TemporalOptions::Date(unit) => **unit,
            TemporalOptions::Timestamp(opts) => opts.unit,
        }
    }

    /// Convert a timestamp value to a Jiff value.
    pub fn to_jiff(&self, v: i64) -> VortexResult<TemporalJiff> {
        match self {
            TemporalOptions::Time(unit) => Ok(TemporalJiff::Time(
                jiff::civil::Time::MIN.checked_add(unit.to_jiff_span(v)?)?,
            )),
            TemporalOptions::Date(unit) => Ok(TemporalJiff::Date(
                jiff::civil::Date::new(1970, 1, 1)?.checked_add(unit.to_jiff_span(v)?)?,
            )),
            TemporalOptions::Timestamp(TimestampOptions { unit, tz }) => match tz {
                None => Ok(TemporalJiff::Unzoned(
                    jiff::civil::DateTime::new(1970, 1, 1, 0, 0, 0, 0)?
                        .checked_add(unit.to_jiff_span(v)?)?,
                )),
                Some(tz) => Ok(TemporalJiff::Zoned(
                    jiff::Timestamp::UNIX_EPOCH
                        .checked_add(unit.to_jiff_span(v)?)?
                        .in_tz(tz.as_ref())?,
                )),
            },
        }
    }
}

/// A Jiff representation of a temporal value.
pub enum TemporalJiff {
    /// A time value.
    Time(jiff::civil::Time),
    /// A date value.
    Date(jiff::civil::Date),
    /// A zone-naive timestamp value.
    Unzoned(jiff::civil::DateTime),
    /// A zoned timestamp value.
    Zoned(jiff::Zoned),
}

impl Display for TemporalJiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemporalJiff::Time(t) => write!(f, "{t}"),
            TemporalJiff::Date(d) => write!(f, "{d}"),
            TemporalJiff::Unzoned(dt) => write!(f, "{dt}"),
            TemporalJiff::Zoned(z) => write!(f, "{z}"),
        }
    }
}
