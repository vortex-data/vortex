// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow conversion logic for Vortex datetime types.
use std::sync::Arc;

use arrow_schema::{DataType, TimeUnit as ArrowTimeUnit};
use vortex_error::{VortexError, VortexExpect as _, VortexResult, vortex_bail, vortex_panic};

use crate::datetime::temporal::{DATE_ID, TIME_ID, TIMESTAMP_ID, TemporalMetadata};
use crate::datetime::unit::TimeUnit;
use crate::{ExtDType, PType};

/// Construct an extension type from the provided temporal Arrow type.
///
/// Supported types are Date32, Date64, Time32, Time64, Timestamp.
pub fn make_temporal_ext_dtype(data_type: &DataType) -> ExtDType {
    assert!(data_type.is_temporal(), "Must receive a temporal DataType");

    match data_type {
        DataType::Timestamp(time_unit, time_zone) => {
            let time_unit = TimeUnit::from(time_unit);
            let tz = time_zone.clone().map(|s| s.to_string());
            // PType is inferred for arrow based on the time units.
            ExtDType::new(
                TIMESTAMP_ID.clone(),
                Arc::new(PType::I64.into()),
                Some(TemporalMetadata::Timestamp(time_unit, tz).into()),
            )
        }
        DataType::Time32(time_unit) => {
            let time_unit = TimeUnit::from(time_unit);
            ExtDType::new(
                TIME_ID.clone(),
                Arc::new(PType::I32.into()),
                Some(TemporalMetadata::Time(time_unit).into()),
            )
        }
        DataType::Time64(time_unit) => {
            let time_unit = TimeUnit::from(time_unit);
            ExtDType::new(
                TIME_ID.clone(),
                Arc::new(PType::I64.into()),
                Some(TemporalMetadata::Time(time_unit).into()),
            )
        }
        DataType::Date32 => ExtDType::new(
            DATE_ID.clone(),
            Arc::new(PType::I32.into()),
            Some(TemporalMetadata::Date(TimeUnit::Days).into()),
        ),
        DataType::Date64 => ExtDType::new(
            DATE_ID.clone(),
            Arc::new(PType::I64.into()),
            Some(TemporalMetadata::Date(TimeUnit::Milliseconds).into()),
        ),
        _ => unimplemented!("{data_type} conversion"),
    }
}

/// Convert temporal ExtDType to a corresponding arrow DataType
///
/// panics if the ext_dtype is not a temporal dtype
pub fn make_arrow_temporal_dtype(ext_dtype: &ExtDType) -> DataType {
    match TemporalMetadata::try_from(ext_dtype)
        .vortex_expect("make_arrow_temporal_dtype must be called with a temporal ExtDType")
    {
        TemporalMetadata::Date(time_unit) => match time_unit {
            TimeUnit::Days => DataType::Date32,
            TimeUnit::Milliseconds => DataType::Date64,
            TimeUnit::Nanoseconds | TimeUnit::Microseconds | TimeUnit::Seconds => {
                vortex_panic!(InvalidArgument: "Invalid TimeUnit {} for {}", time_unit, ext_dtype.id())
            }
        },
        TemporalMetadata::Time(time_unit) => match time_unit {
            TimeUnit::Seconds => DataType::Time32(ArrowTimeUnit::Second),
            TimeUnit::Milliseconds => DataType::Time32(ArrowTimeUnit::Millisecond),
            TimeUnit::Microseconds => DataType::Time64(ArrowTimeUnit::Microsecond),
            TimeUnit::Nanoseconds => DataType::Time64(ArrowTimeUnit::Nanosecond),
            TimeUnit::Days => {
                vortex_panic!(InvalidArgument: "Invalid TimeUnit {} for {}", time_unit, ext_dtype.id())
            }
        },
        TemporalMetadata::Timestamp(time_unit, tz) => match time_unit {
            TimeUnit::Nanoseconds => {
                DataType::Timestamp(ArrowTimeUnit::Nanosecond, tz.map(|t| t.into()))
            }
            TimeUnit::Microseconds => {
                DataType::Timestamp(ArrowTimeUnit::Microsecond, tz.map(|t| t.into()))
            }
            TimeUnit::Milliseconds => {
                DataType::Timestamp(ArrowTimeUnit::Millisecond, tz.map(|t| t.into()))
            }
            TimeUnit::Seconds => DataType::Timestamp(ArrowTimeUnit::Second, tz.map(|t| t.into())),
            TimeUnit::Days => {
                vortex_panic!(InvalidArgument: "Invalid TimeUnit {} for {}", time_unit, ext_dtype.id())
            }
        },
    }
}

impl From<&ArrowTimeUnit> for TimeUnit {
    fn from(value: &ArrowTimeUnit) -> Self {
        (*value).into()
    }
}

impl From<ArrowTimeUnit> for TimeUnit {
    fn from(value: ArrowTimeUnit) -> Self {
        match value {
            ArrowTimeUnit::Second => Self::Seconds,
            ArrowTimeUnit::Millisecond => Self::Milliseconds,
            ArrowTimeUnit::Microsecond => Self::Microseconds,
            ArrowTimeUnit::Nanosecond => Self::Nanoseconds,
        }
    }
}

impl TryFrom<TimeUnit> for ArrowTimeUnit {
    type Error = VortexError;

    fn try_from(value: TimeUnit) -> VortexResult<Self> {
        Ok(match value {
            TimeUnit::Seconds => Self::Second,
            TimeUnit::Milliseconds => Self::Millisecond,
            TimeUnit::Microseconds => Self::Microsecond,
            TimeUnit::Nanoseconds => Self::Nanosecond,
            _ => vortex_bail!("Cannot convert {value} to Arrow TimeUnit"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_arrow_timestamp() {
        let ext_dtype = ExtDType::new(
            TIMESTAMP_ID.clone(),
            Arc::new(PType::I64.into()),
            Some(TemporalMetadata::Timestamp(TimeUnit::Milliseconds, None).into()),
        );
        let expected_arrow_type = DataType::Timestamp(ArrowTimeUnit::Millisecond, None);

        let arrow_dtype = make_arrow_temporal_dtype(&ext_dtype);
        assert_eq!(arrow_dtype, expected_arrow_type);

        let rt_ext_dtype = make_temporal_ext_dtype(&expected_arrow_type);
        assert_eq!(ext_dtype, rt_ext_dtype);
    }

    #[test]
    fn test_make_arrow_time32() {
        let ext_dtype = ExtDType::new(
            TIME_ID.clone(),
            Arc::new(PType::I32.into()),
            Some(TemporalMetadata::Time(TimeUnit::Milliseconds).into()),
        );
        let expected_arrow_type = DataType::Time32(ArrowTimeUnit::Millisecond);
        let arrow_dtype = make_arrow_temporal_dtype(&ext_dtype);
        assert_eq!(arrow_dtype, expected_arrow_type);

        let rt_ext_dtype = make_temporal_ext_dtype(&expected_arrow_type);
        assert_eq!(ext_dtype, rt_ext_dtype);
    }

    #[test]
    fn test_make_arrow_time64() {
        let ext_dtype = ExtDType::new(
            TIME_ID.clone(),
            Arc::new(PType::I64.into()),
            Some(TemporalMetadata::Time(TimeUnit::Microseconds).into()),
        );
        let expected_arrow_type = DataType::Time64(ArrowTimeUnit::Microsecond);
        let arrow_dtype = make_arrow_temporal_dtype(&ext_dtype);
        assert_eq!(arrow_dtype, expected_arrow_type);

        let rt_ext_dtype = make_temporal_ext_dtype(&expected_arrow_type);
        assert_eq!(ext_dtype, rt_ext_dtype);
    }

    #[test]
    fn test_make_arrow_date32() {
        let ext_dtype = ExtDType::new(
            DATE_ID.clone(),
            Arc::new(PType::I32.into()),
            Some(TemporalMetadata::Date(TimeUnit::Days).into()),
        );
        let expected_arrow_type = DataType::Date32;
        let arrow_dtype = make_arrow_temporal_dtype(&ext_dtype);
        assert_eq!(arrow_dtype, expected_arrow_type);

        let rt_ext_dtype = make_temporal_ext_dtype(&expected_arrow_type);
        assert_eq!(ext_dtype, rt_ext_dtype);
    }

    #[test]
    fn test_make_arrow_date64() {
        let ext_dtype = ExtDType::new(
            DATE_ID.clone(),
            Arc::new(PType::I64.into()),
            Some(TemporalMetadata::Date(TimeUnit::Milliseconds).into()),
        );
        let expected_arrow_type = DataType::Date64;
        let arrow_dtype = make_arrow_temporal_dtype(&ext_dtype);
        assert_eq!(arrow_dtype, expected_arrow_type);

        let rt_ext_dtype = make_temporal_ext_dtype(&expected_arrow_type);
        assert_eq!(ext_dtype, rt_ext_dtype);
    }
}
