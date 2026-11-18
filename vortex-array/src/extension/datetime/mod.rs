// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Datetime extension DTypes, compatible with Apache Arrow.

mod date;
mod matcher;
mod time;
mod timestamp;
mod unit;

pub use date::*;
pub use matcher::*;
pub use time::*;
pub use timestamp::*;
pub use unit::*;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::Date;
    use super::Time;
    use super::TimeUnit;
    use super::Timestamp;
    use crate::dtype::Claim;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::extension::ExtDTypeRef;

    #[rstest]
    #[case::date_days(Date::new(TimeUnit::Days, NonNullable).erased())]
    #[case::date_milliseconds(Date::new(TimeUnit::Milliseconds, NonNullable).erased())]
    #[case::time_seconds(Time::new(TimeUnit::Seconds, NonNullable).erased())]
    #[case::time_milliseconds(Time::new(TimeUnit::Milliseconds, NonNullable).erased())]
    #[case::time_microseconds(Time::new(TimeUnit::Microseconds, NonNullable).erased())]
    #[case::time_nanoseconds(Time::new(TimeUnit::Nanoseconds, NonNullable).erased())]
    #[case::timestamp_milliseconds(Timestamp::new(TimeUnit::Milliseconds, NonNullable).erased())]
    #[case::timestamp_seconds_utc(
        Timestamp::new_with_tz(TimeUnit::Seconds, Some("UTC".into()), NonNullable).erased()
    )]
    #[case::timestamp_microseconds_iana(Timestamp::new_with_tz(
        TimeUnit::Microseconds, Some("America/New_York".into()), NonNullable,
    ).erased())]
    #[case::timestamp_nanoseconds_offset(
        Timestamp::new_with_tz(TimeUnit::Nanoseconds, Some("+05:30".into()), NonNullable).erased()
    )]
    fn temporal_types_claim_ord(
        #[case] ext_dtype: ExtDTypeRef,
        #[values(NonNullable, Nullable)] nullability: Nullability,
    ) {
        let dtype = DType::Extension(ext_dtype.with_nullability(nullability));

        assert!(dtype.holds(Claim::Ord), "{dtype} must hold Ord");
        assert!(dtype.holds(Claim::Eq), "{dtype} must hold Eq");
    }
}
