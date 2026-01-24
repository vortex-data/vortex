// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::datetime::Date;
use crate::datetime::Time;
use crate::datetime::Timestamp;
use crate::extension::ExtDTypeRef;
use crate::extension::Matcher;
use crate::extension::VTable;

/// Options for temporal extension data types.
pub enum TemporalOptions<'a> {
    /// Options for Timestamp dtypes
    Timestamp(&'a <Timestamp as VTable>::Options),
    /// Options for Date dtypes
    Date(&'a <Date as VTable>::Options),
    /// Options for Time dtypes
    Time(&'a <Time as VTable>::Options),
}

/// Matcher for temporal extension data types.
pub struct AnyTemporal;

impl Matcher for AnyTemporal {
    type Match<'a> = TemporalOptions<'a>;

    fn try_match<'a>(item: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if let Some(opts) = item.try_options::<Timestamp>() {
            return Some(TemporalOptions::Timestamp(opts));
        }
        if let Some(opts) = item.try_options::<Time>() {
            return Some(TemporalOptions::Date(opts));
        }
        if let Some(opts) = item.try_options::<Time>() {
            return Some(TemporalOptions::Time(opts));
        }
        None
    }
}
