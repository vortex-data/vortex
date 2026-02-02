// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::sync::Arc;

use vortex_error::VortexResult;

use crate::datetime::Date;
use crate::datetime::Time;
use crate::datetime::TimeUnit;
use crate::datetime::Timestamp;
use crate::extension::ExtDTypeRef;
use crate::extension::ExtDTypeVTable;
use crate::extension::Matcher;

/// Matcher for temporal extension data types.
pub struct AnyTemporal;

impl Matcher for AnyTemporal {
    type Match<'a> = TemporalMetadata<'a>;

    fn try_match<'a>(item: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if let Some(opts) = item.metadata_opt::<Timestamp>() {
            return Some(TemporalMetadata::Timestamp(&opts.unit, &opts.tz));
        }
        if let Some(opts) = item.metadata_opt::<Date>() {
            return Some(TemporalMetadata::Date(opts));
        }
        if let Some(opts) = item.metadata_opt::<Time>() {
            return Some(TemporalMetadata::Time(opts));
        }
        None
    }
}

/// Metadata for temporal extension data types.
#[derive(Debug, PartialEq, Eq)]
pub enum TemporalMetadata<'a> {
    /// Metadata for Timestamp dtypes, a tuple of time unit and optional timezone.
    Timestamp(&'a TimeUnit, &'a Option<Arc<str>>),
    /// Metadata for Date dtypes
    Date(&'a <Date as ExtDTypeVTable>::Metadata),
    /// Metadata for Time dtypes
    Time(&'a <Time as ExtDTypeVTable>::Metadata),
}

// TODO(ngates): remove this logic in favor of having an ExtScalarVTable in vortex-scalar.
//  Currently this is used largely to implement scalar display hacks.
impl TemporalMetadata<'_> {
    /// Get the time unit of the temporal dtype.
    pub fn time_unit(&self) -> TimeUnit {
        match self {
            TemporalMetadata::Time(unit) => **unit,
            TemporalMetadata::Date(unit) => **unit,
            TemporalMetadata::Timestamp(unit, _tz) => **unit,
        }
    }

    /// Convert a timestamp value to a Jiff value.
    pub fn to_jiff(&self, v: i64) -> VortexResult<TemporalJiff> {
        match self {
            TemporalMetadata::Time(unit) => Ok(TemporalJiff::Time(
                jiff::civil::Time::MIN.checked_add(unit.to_jiff_span(v)?)?,
            )),
            TemporalMetadata::Date(unit) => Ok(TemporalJiff::Date(
                jiff::civil::Date::new(1970, 1, 1)?.checked_add(unit.to_jiff_span(v)?)?,
            )),
            TemporalMetadata::Timestamp(unit, tz) => match tz {
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
