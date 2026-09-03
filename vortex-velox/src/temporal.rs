// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_data::ArrayData;
use arrow_schema::DataType;
use arrow_schema::TimeUnit;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

const VELOX_TIMESTAMP_MAX_SECONDS: i64 = i64::MAX / 1_000;
const VELOX_TIMESTAMP_MIN_SECONDS: i64 = i64::MIN / 1_000 - 1;

pub(crate) fn validate_velox_arrow_type(data_type: &DataType) -> VortexResult<()> {
    match data_type {
        DataType::Timestamp(_, Some(timezone)) => {
            vortex_bail!(
                "Velox Vortex scans do not support timestamp timezone metadata: {timezone}"
            )
        }
        DataType::List(field)
        | DataType::ListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::LargeList(field)
        | DataType::LargeListView(field)
        | DataType::Map(field, _) => validate_velox_arrow_type(field.data_type()),
        DataType::Struct(fields) => {
            for field in fields {
                validate_velox_arrow_type(field.data_type())?;
            }
            Ok(())
        }
        DataType::Union(fields, _) => {
            for (_, field) in fields.iter() {
                validate_velox_arrow_type(field.data_type())?;
            }
            Ok(())
        }
        DataType::Dictionary(_, values) => validate_velox_arrow_type(values),
        DataType::RunEndEncoded(_, values) => validate_velox_arrow_type(values.data_type()),
        _ => Ok(()),
    }
}

pub(crate) fn validate_velox_arrow_data(data: &ArrayData) -> VortexResult<()> {
    validate_velox_arrow_type(data.data_type())?;
    if matches!(
        data.data_type(),
        DataType::Timestamp(TimeUnit::Second, None)
    ) {
        let values = data
            .buffers()
            .first()
            .ok_or_else(|| vortex_err!("Arrow timestamp array lacks a values buffer"))?
            .typed_data::<i64>();
        let end = data
            .offset()
            .checked_add(data.len())
            .ok_or_else(|| vortex_err!("Arrow timestamp array range overflows"))?;
        let values = values.get(data.offset()..end).ok_or_else(|| {
            vortex_err!(
                "Arrow timestamp values buffer is too small: expected {}, got {}",
                end,
                values.len()
            )
        })?;
        for (index, value) in values.iter().enumerate() {
            if data.nulls().is_some_and(|nulls| !nulls.is_valid(index)) {
                continue;
            }
            if !(VELOX_TIMESTAMP_MIN_SECONDS..=VELOX_TIMESTAMP_MAX_SECONDS).contains(value) {
                vortex_bail!("Timestamp seconds exceed the Velox range: {value}");
            }
        }
    }
    for child in data.child_data() {
        validate_velox_arrow_data(child)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array;
    use arrow_array::ArrayRef;
    use arrow_array::StructArray;
    use arrow_array::TimestampMillisecondArray;
    use arrow_array::TimestampSecondArray;
    use arrow_buffer::NullBuffer;
    use arrow_buffer::ScalarBuffer;
    use arrow_schema::Field;
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn rejects_nested_timestamp_timezone_metadata() -> VortexResult<()> {
        let data_type = DataType::Struct(
            vec![Field::new(
                "timestamp",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            )]
            .into(),
        );
        let error = match validate_velox_arrow_type(&data_type) {
            Ok(()) => vortex_bail!("Timestamp timezone metadata unexpectedly passed validation"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("timezone metadata: UTC"));
        Ok(())
    }

    #[test]
    fn validates_nested_second_timestamp_range_and_nulls() -> VortexResult<()> {
        let valid = Arc::new(TimestampSecondArray::from(vec![
            Some(VELOX_TIMESTAMP_MIN_SECONDS),
            None,
            Some(VELOX_TIMESTAMP_MAX_SECONDS),
        ])) as ArrayRef;
        let valid = StructArray::from(vec![(
            Arc::new(Field::new("timestamp", valid.data_type().clone(), true)),
            valid,
        )]);
        validate_velox_arrow_data(&valid.to_data())?;

        let invalid = TimestampSecondArray::from(vec![Some(VELOX_TIMESTAMP_MAX_SECONDS + 1), None]);
        let error = match validate_velox_arrow_data(&invalid.to_data()) {
            Ok(()) => vortex_bail!("Out-of-range timestamp unexpectedly passed validation"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("exceed the Velox range"));

        let null_out_of_range = TimestampSecondArray::new(
            ScalarBuffer::from(vec![VELOX_TIMESTAMP_MAX_SECONDS + 1]),
            Some(NullBuffer::new_null(1)),
        )
        .to_data();
        validate_velox_arrow_data(&null_out_of_range)?;
        Ok(())
    }

    #[test]
    fn accepts_full_millisecond_storage_range() -> VortexResult<()> {
        let timestamps = TimestampMillisecondArray::from(vec![i64::MIN, i64::MAX]);
        validate_velox_arrow_data(&timestamps.to_data())
    }
}
