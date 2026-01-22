// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow conversion logic for Vortex datetime types.
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::TimeUnit as ArrowTimeUnit;
use vortex_error::VortexError;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ExtDType;
use crate::PType;
use crate::datetime::temporal::TemporalMetadata;
use crate::datetime::unit::TimeUnit;

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
