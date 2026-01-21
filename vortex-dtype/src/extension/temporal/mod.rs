// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod date;
mod time;
mod timestamp;

pub use date::*;
pub use time::*;
pub use timestamp::*;

use crate::datetime::TemporalMetadata;
use crate::v2::ExtDTypeRef;
use crate::v2::Matcher;

/// Matcher for temporal extension data types.
pub struct AnyTemporal;

impl Matcher for AnyTemporal {
    type Match<'a> = TemporalMetadata;

    fn try_match<'a>(item: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if let Some(opts) = item.try_options::<Timestamp>() {
            return Some(TemporalMetadata::Timestamp(opts.unit, opts.tz.clone()));
        }
        if let Some(time_unit) = item.try_options::<Time>() {
            return Some(TemporalMetadata::Date(*time_unit));
        }

        // FIXME(ngate): time
        None
    }
}
