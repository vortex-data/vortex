// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::str::FromStr;

use prost::Message;
use vortex_buffer::Buffer;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::dtype::DType;
use crate::dtype::extension::ExtDTypeRef;
use crate::expr::Expression;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::proto::expr as pb;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// The calendar unit a [`DateTrunc`] expression truncates timestamps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DateTruncUnit {
    Microsecond,
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    /// Weeks start on Monday, matching PostgreSQL, DataFusion and DuckDB.
    Week,
    Month,
    Quarter,
    Year,
}

impl DateTruncUnit {
    /// Returns the lowercase SQL name of this unit, e.g. `month`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Microsecond => "microsecond",
            Self::Millisecond => "millisecond",
            Self::Second => "second",
            Self::Minute => "minute",
            Self::Hour => "hour",
            Self::Day => "day",
            Self::Week => "week",
            Self::Month => "month",
            Self::Quarter => "quarter",
            Self::Year => "year",
        }
    }
}

impl Display for DateTruncUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for DateTruncUnit {
    type Err = VortexError;

    /// Parses a SQL `DATE_TRUNC` precision, case-insensitively.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "microsecond" => Self::Microsecond,
            "millisecond" => Self::Millisecond,
            "second" => Self::Second,
            "minute" => Self::Minute,
            "hour" => Self::Hour,
            "day" => Self::Day,
            "week" => Self::Week,
            "month" => Self::Month,
            "quarter" => Self::Quarter,
            "year" => Self::Year,
            other => vortex_bail!("Unsupported date_trunc unit: {other}"),
        })
    }
}

impl From<DateTruncUnit> for pb::date_trunc_opts::DateTruncUnit {
    fn from(value: DateTruncUnit) -> Self {
        match value {
            DateTruncUnit::Microsecond => Self::Microsecond,
            DateTruncUnit::Millisecond => Self::Millisecond,
            DateTruncUnit::Second => Self::Second,
            DateTruncUnit::Minute => Self::Minute,
            DateTruncUnit::Hour => Self::Hour,
            DateTruncUnit::Day => Self::Day,
            DateTruncUnit::Week => Self::Week,
            DateTruncUnit::Month => Self::Month,
            DateTruncUnit::Quarter => Self::Quarter,
            DateTruncUnit::Year => Self::Year,
        }
    }
}

impl From<pb::date_trunc_opts::DateTruncUnit> for DateTruncUnit {
    fn from(value: pb::date_trunc_opts::DateTruncUnit) -> Self {
        use pb::date_trunc_opts::DateTruncUnit as Pb;
        match value {
            Pb::Microsecond => Self::Microsecond,
            Pb::Millisecond => Self::Millisecond,
            Pb::Second => Self::Second,
            Pb::Minute => Self::Minute,
            Pb::Hour => Self::Hour,
            Pb::Day => Self::Day,
            Pb::Week => Self::Week,
            Pb::Month => Self::Month,
            Pb::Quarter => Self::Quarter,
            Pb::Year => Self::Year,
        }
    }
}

/// Returns true if `tz` names UTC, where calendar truncation needs no offset handling.
pub fn is_utc_timezone(tz: &str) -> bool {
    matches!(
        tz.to_ascii_lowercase().as_str(),
        "utc" | "etc/utc" | "z" | "+00:00" | "-00:00" | "+00" | "-00" | "+0000" | "-0000"
    )
}

/// Truncates each timestamp down to the start of a calendar unit, like SQL `DATE_TRUNC`.
///
/// The input must be a [`Timestamp`] extension array that is either timezone-naive or in UTC;
/// truncation happens on the UTC calendar. The output has the same dtype as the input.
/// Truncating to a unit finer than the timestamp's resolution is the identity.
#[derive(Clone)]
pub struct DateTrunc;

impl ScalarFnVTable for DateTrunc {
    type Options = DateTruncUnit;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.date_trunc");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::DateTruncOpts {
                unit: pb::date_trunc_opts::DateTruncUnit::from(*options).into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::DateTruncOpts::decode(metadata)?;
        Ok(pb::date_trunc_opts::DateTruncUnit::try_from(opts.unit)?.into())
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for date_trunc()"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        timestamp_unit(&arg_dtypes[0])?;
        Ok(arg_dtypes[0].clone())
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let time_unit = timestamp_unit(input.dtype())?;
        let Some(divisor) = TruncDivisor::new(*options, time_unit) else {
            // The unit is at or below the timestamp resolution, so there is nothing to strip.
            return Ok(input);
        };
        let ext_dtype = input.dtype().as_extension().clone();

        if let Some(scalar) = input.as_constant() {
            let truncated = scalar_date_trunc(&scalar, &ext_dtype, divisor)?;
            return Ok(ConstantArray::new(truncated, args.row_count()).into_array());
        }

        let storage = input
            .execute::<ExtensionArray>(ctx)?
            .storage_array()
            .clone()
            .execute::<PrimitiveArray>(ctx)?;
        let validity = storage.validity()?;
        let values = storage
            .as_slice::<i64>()
            .iter()
            .map(|v| divisor.truncate(*v))
            .collect::<VortexResult<Buffer<i64>>>()?;
        let storage = PrimitiveArray::new(values, validity).into_array();
        Ok(ExtensionArray::new(ext_dtype, storage).into_array())
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(expression.child(0).validity()?))
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Returns the storage resolution of a UTC or timezone-naive [`Timestamp`] dtype.
fn timestamp_unit(dtype: &DType) -> VortexResult<TimeUnit> {
    let options = dtype
        .as_extension_opt()
        .and_then(|ext| ext.metadata_opt::<Timestamp>())
        .ok_or_else(|| vortex_err!("date_trunc() requires a Timestamp, got {dtype}"))?;
    if let Some(tz) = &options.tz
        && !is_utc_timezone(tz)
    {
        vortex_bail!("date_trunc() only supports UTC or timezone-naive timestamps, got tz={tz}");
    }
    if options.unit == TimeUnit::Days {
        vortex_bail!("date_trunc() does not support timestamps with day resolution");
    }
    Ok(options.unit)
}

fn scalar_date_trunc(
    scalar: &Scalar,
    ext_dtype: &ExtDTypeRef,
    divisor: TruncDivisor,
) -> VortexResult<Scalar> {
    if scalar.is_null() {
        return Ok(scalar.clone());
    }
    let storage = scalar.as_extension().to_storage_scalar();
    let value = storage
        .as_primitive()
        .typed_value::<i64>()
        .vortex_expect("non-null timestamp storage");
    let truncated = Scalar::primitive(divisor.truncate(value)?, storage.dtype().nullability());
    Ok(Scalar::extension_ref(ext_dtype.clone(), truncated))
}

/// How to truncate a raw timestamp value, in the storage resolution of the array.
#[derive(Clone, Copy)]
enum TruncDivisor {
    /// Floor to a multiple of a fixed number of storage units. Covers every unit up to a day.
    Fixed(i64),
    /// Calendar units, which are not a fixed number of days long. Carries the number of
    /// storage units per day.
    Calendar(DateTruncUnit, i64),
}

impl TruncDivisor {
    /// Returns `None` when the unit is at or below the storage resolution, i.e. the identity.
    fn new(unit: DateTruncUnit, time_unit: TimeUnit) -> Option<Self> {
        let per_second: i64 = match time_unit {
            TimeUnit::Nanoseconds => 1_000_000_000,
            TimeUnit::Microseconds => 1_000_000,
            TimeUnit::Milliseconds => 1_000,
            TimeUnit::Seconds => 1,
            TimeUnit::Days => return None,
        };
        let per_day = per_second * 86_400;
        let fixed = |n: i64| (n > 1).then_some(Self::Fixed(n));
        match unit {
            DateTruncUnit::Microsecond => fixed(per_second / 1_000_000),
            DateTruncUnit::Millisecond => fixed(per_second / 1_000),
            DateTruncUnit::Second => fixed(per_second),
            DateTruncUnit::Minute => fixed(per_second * 60),
            DateTruncUnit::Hour => fixed(per_second * 3_600),
            DateTruncUnit::Day => fixed(per_day),
            DateTruncUnit::Week
            | DateTruncUnit::Month
            | DateTruncUnit::Quarter
            | DateTruncUnit::Year => Some(Self::Calendar(unit, per_day)),
        }
    }

    /// Floors `value` to the start of the unit. Only fails when the result would fall below
    /// `i64::MIN`, which no real timestamp gets close to.
    fn truncate(self, value: i64) -> VortexResult<i64> {
        let out_of_range = || vortex_err!("Timestamp {value} out of range after date_trunc()");
        match self {
            Self::Fixed(n) => value
                .checked_sub(value.rem_euclid(n))
                .ok_or_else(out_of_range),
            Self::Calendar(unit, per_day) => {
                let days = value.div_euclid(per_day);
                let start = match unit {
                    // The epoch is a Thursday, three days after a Monday.
                    DateTruncUnit::Week => days - (days + 3).rem_euclid(7),
                    DateTruncUnit::Month => {
                        let (_, _, day_of_month) = civil_from_days(days);
                        days - (day_of_month - 1)
                    }
                    DateTruncUnit::Quarter => {
                        let (year, month, _) = civil_from_days(days);
                        days_from_civil(year, 1 + 3 * ((month - 1) / 3), 1)
                    }
                    DateTruncUnit::Year => {
                        let (year, ..) = civil_from_days(days);
                        days_from_civil(year, 1, 1)
                    }
                    _ => unreachable!("{unit} is a fixed-length unit"),
                };
                start.checked_mul(per_day).ok_or_else(out_of_range)
            }
        }
    }
}

/// Days from the Unix epoch to 0000-03-01, the epoch of the March-based calendar below.
const DAYS_EPOCH_SHIFT: i64 = 719_468;

/// Days in a 400 year era of the proleptic Gregorian calendar.
const DAYS_PER_ERA: i64 = 146_097;

/// Splits a day count relative to the Unix epoch into a proleptic Gregorian year, month (1-12)
/// and day of month (1-31).
///
/// A port of Howard Hinnant's `civil_from_days`, which derives the constants:
/// <https://howardhinnant.github.io/date_algorithms.html#civil_from_days>
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + DAYS_EPOCH_SHIFT;
    let era = z.div_euclid(DAYS_PER_ERA);
    let day_of_era = z.rem_euclid(DAYS_PER_ERA);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    // Month index with March as 0, so the leap day falls at the end of the year.
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Inverse of [`civil_from_days`]: the day count relative to the Unix epoch of a date.
///
/// A port of Howard Hinnant's `days_from_civil`:
/// <https://howardhinnant.github.io/date_algorithms.html#days_from_civil>
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year.rem_euclid(400);
    let month_index = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * month_index + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * DAYS_PER_ERA + day_of_era - DAYS_EPOCH_SHIFT
}

#[cfg(test)]
mod tests {
    use jiff::civil::DateTime;
    use jiff::civil::date;
    use jiff::tz::TimeZone;
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::datetime::TemporalData;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability;
    use crate::expr::date_trunc;
    use crate::expr::root;

    /// Microseconds since the epoch of a civil datetime in UTC.
    fn micros(dt: DateTime) -> i64 {
        dt.to_zoned(TimeZone::UTC)
            .vortex_expect("valid datetime")
            .timestamp()
            .as_microsecond()
    }

    fn timestamps(values: Vec<i64>, unit: TimeUnit) -> ArrayRef {
        TemporalData::new_timestamp(PrimitiveArray::from_iter(values).into_array(), unit, None)
            .into_array()
    }

    fn nullable_timestamps(values: Vec<Option<i64>>, unit: TimeUnit) -> ArrayRef {
        TemporalData::new_timestamp(
            PrimitiveArray::from_option_iter(values).into_array(),
            unit,
            None,
        )
        .into_array()
    }

    const SAMPLE: DateTime = date(2013, 7, 14).at(12, 34, 56, 789_123_000);
    // A Sunday, so the week starts on the previous Monday.
    const SUNDAY: DateTime = date(2024, 3, 3).at(23, 59, 59, 999_999_000);
    // A leap day in Q1, before the epoch.
    const LEAP_DAY: DateTime = date(1968, 2, 29).at(6, 0, 0, 0);

    #[rstest]
    #[case(DateTruncUnit::Microsecond, SAMPLE, date(2013, 7, 14).at(12, 34, 56, 789_123_000))]
    #[case(DateTruncUnit::Millisecond, SAMPLE, date(2013, 7, 14).at(12, 34, 56, 789_000_000))]
    #[case(DateTruncUnit::Second, SAMPLE, date(2013, 7, 14).at(12, 34, 56, 0))]
    #[case(DateTruncUnit::Minute, SAMPLE, date(2013, 7, 14).at(12, 34, 0, 0))]
    #[case(DateTruncUnit::Hour, SAMPLE, date(2013, 7, 14).at(12, 0, 0, 0))]
    #[case(DateTruncUnit::Day, SAMPLE, date(2013, 7, 14).at(0, 0, 0, 0))]
    #[case(DateTruncUnit::Week, SAMPLE, date(2013, 7, 8).at(0, 0, 0, 0))]
    #[case(DateTruncUnit::Month, SAMPLE, date(2013, 7, 1).at(0, 0, 0, 0))]
    #[case(DateTruncUnit::Quarter, SAMPLE, date(2013, 7, 1).at(0, 0, 0, 0))]
    #[case(DateTruncUnit::Year, SAMPLE, date(2013, 1, 1).at(0, 0, 0, 0))]
    #[case(DateTruncUnit::Week, SUNDAY, date(2024, 2, 26).at(0, 0, 0, 0))]
    #[case(DateTruncUnit::Month, SUNDAY, date(2024, 3, 1).at(0, 0, 0, 0))]
    #[case(DateTruncUnit::Quarter, SUNDAY, date(2024, 1, 1).at(0, 0, 0, 0))]
    #[case(DateTruncUnit::Day, LEAP_DAY, date(1968, 2, 29).at(0, 0, 0, 0))]
    #[case(DateTruncUnit::Week, LEAP_DAY, date(1968, 2, 26).at(0, 0, 0, 0))]
    #[case(DateTruncUnit::Month, LEAP_DAY, date(1968, 2, 1).at(0, 0, 0, 0))]
    #[case(DateTruncUnit::Quarter, LEAP_DAY, date(1968, 1, 1).at(0, 0, 0, 0))]
    #[case(DateTruncUnit::Year, LEAP_DAY, date(1968, 1, 1).at(0, 0, 0, 0))]
    fn test_date_trunc_micros(
        #[case] unit: DateTruncUnit,
        #[case] input: DateTime,
        #[case] expected: DateTime,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = timestamps(vec![micros(input)], TimeUnit::Microseconds);
        let result = array.apply(&date_trunc(unit, root()))?;
        let expected = timestamps(vec![micros(expected)], TimeUnit::Microseconds);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(TimeUnit::Seconds, 1, DateTruncUnit::Minute, 60)]
    #[case(TimeUnit::Milliseconds, 1_000, DateTruncUnit::Hour, 3_600_000)]
    #[case(
        TimeUnit::Nanoseconds,
        1_000_000_000,
        DateTruncUnit::Millisecond,
        1_000_000
    )]
    #[case(
        TimeUnit::Nanoseconds,
        1_000_000_000,
        DateTruncUnit::Day,
        86_400_000_000_000
    )]
    fn test_date_trunc_other_units(
        #[case] time_unit: TimeUnit,
        #[case] per_second: i64,
        #[case] unit: DateTruncUnit,
        #[case] step: i64,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // A day and change after the epoch, plus a value just before it.
        let base = 90_000 * per_second + 7;
        let array = timestamps(vec![base, -1], time_unit);
        let result = array.apply(&date_trunc(unit, root()))?;
        let expected = timestamps(vec![base - base.rem_euclid(step), -step], time_unit);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_finer_than_resolution_is_identity() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = timestamps(vec![1_234_567, -89], TimeUnit::Seconds);
        let result = array
            .clone()
            .apply(&date_trunc(DateTruncUnit::Millisecond, root()))?;
        assert_arrays_eq!(result, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_nullable_date_trunc() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = nullable_timestamps(vec![Some(micros(SAMPLE)), None], TimeUnit::Microseconds);
        let result = array.apply(&date_trunc(DateTruncUnit::Month, root()))?;
        let expected = nullable_timestamps(
            vec![Some(micros(date(2013, 7, 1).at(0, 0, 0, 0))), None],
            TimeUnit::Microseconds,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_constant_date_trunc() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = Timestamp::new(TimeUnit::Microseconds, Nullability::NonNullable);
        let scalar = Scalar::extension_ref(
            dtype.erased(),
            Scalar::primitive(micros(SAMPLE), Nullability::NonNullable),
        );
        let array = ConstantArray::new(scalar, 3).into_array();
        let result = array.apply(&date_trunc(DateTruncUnit::Year, root()))?;
        let expected = timestamps(
            vec![micros(date(2013, 1, 1).at(0, 0, 0, 0)); 3],
            TimeUnit::Microseconds,
        );
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_rejects_non_utc_timezone() {
        let dtype = DType::Extension(
            Timestamp::new_with_tz(
                TimeUnit::Microseconds,
                Some("America/New_York".into()),
                Nullability::NonNullable,
            )
            .erased(),
        );
        assert!(
            DateTrunc
                .return_dtype(&DateTruncUnit::Day, &[dtype])
                .is_err()
        );

        let utc = DType::Extension(
            Timestamp::new_with_tz(
                TimeUnit::Microseconds,
                Some("UTC".into()),
                Nullability::NonNullable,
            )
            .erased(),
        );
        assert_eq!(
            DateTrunc
                .return_dtype(&DateTruncUnit::Day, std::slice::from_ref(&utc))
                .unwrap(),
            utc
        );
    }

    #[test]
    fn test_parse_unit() {
        assert_eq!(
            "MONTH".parse::<DateTruncUnit>().unwrap(),
            DateTruncUnit::Month
        );
        assert!("fortnight".parse::<DateTruncUnit>().is_err());
    }

    #[test]
    fn test_serde_round_trip() -> VortexResult<()> {
        let bytes = DateTrunc
            .serialize(&DateTruncUnit::Quarter)?
            .vortex_expect("serializable");
        let unit = DateTrunc.deserialize(&bytes, &array_session())?;
        assert_eq!(unit, DateTruncUnit::Quarter);
        Ok(())
    }

    #[test]
    fn test_display() {
        let expr = date_trunc(DateTruncUnit::Month, root());
        assert_eq!(expr.to_string(), "vortex.date_trunc($, opts=month)");
    }
}
