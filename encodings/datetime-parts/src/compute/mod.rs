mod filter;
mod take;

use vortex_array::array::{PrimitiveArray, TemporalArray};
use vortex_array::compute::{
    scalar_at, slice, try_cast, ComputeVTable, FilterFn, ScalarAtFn, SliceFn, TakeFn,
};
use vortex_array::validity::ArrayValidity;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_datetime_dtype::{TemporalMetadata, TimeUnit};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_scalar::{PrimitiveScalar, Scalar};

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl ComputeVTable for DateTimePartsEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}

impl SliceFn<DateTimePartsArray> for DateTimePartsEncoding {
    fn slice(
        &self,
        array: &DateTimePartsArray,
        start: usize,
        stop: usize,
    ) -> VortexResult<ArrayData> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            slice(array.days(), start, stop)?,
            slice(array.seconds(), start, stop)?,
            slice(array.subsecond(), start, stop)?,
        )?
        .into_array())
    }
}

impl ScalarAtFn<DateTimePartsArray> for DateTimePartsEncoding {
    fn scalar_at(&self, array: &DateTimePartsArray, index: usize) -> VortexResult<Scalar> {
        let DType::Extension(ext) = array.dtype().clone() else {
            vortex_bail!(
                "DateTimePartsArray must have extension dtype, found {}",
                array.dtype()
            );
        };

        let TemporalMetadata::Timestamp(time_unit, _) = TemporalMetadata::try_from(ext.as_ref())?
        else {
            vortex_bail!("Metadata must be Timestamp, found {}", ext.id());
        };

        if !array.is_valid(index) {
            return Ok(Scalar::null(DType::Extension(ext)));
        }

        let divisor = match time_unit {
            TimeUnit::Ns => 1_000_000_000,
            TimeUnit::Us => 1_000_000,
            TimeUnit::Ms => 1_000,
            TimeUnit::S => 1,
            TimeUnit::D => vortex_bail!("Invalid time unit D"),
        };

        let days: i64 = scalar_at(array.days(), index)?.try_into()?;
        let seconds: i64 = scalar_at(array.seconds(), index)?.try_into()?;
        let subseconds: i64 = scalar_at(array.subsecond(), index)?.try_into()?;

        let scalar = days * 86_400 * divisor + seconds * divisor + subseconds;

        Ok(Scalar::extension(ext, Scalar::from(scalar)))
    }
}

/// Decode an [ArrayData] into a [TemporalArray].
///
/// Enforces that the passed array is actually a [DateTimePartsArray] with proper metadata.
pub fn decode_to_temporal(array: &DateTimePartsArray) -> VortexResult<TemporalArray> {
    let DType::Extension(ext) = array.dtype().clone() else {
        vortex_bail!(ComputeError: "expected dtype to be DType::Extension variant")
    };

    let Ok(temporal_metadata) = TemporalMetadata::try_from(ext.as_ref()) else {
        vortex_bail!(ComputeError: "must decode TemporalMetadata from extension metadata");
    };

    let divisor = match temporal_metadata.time_unit() {
        TimeUnit::Ns => 1_000_000_000,
        TimeUnit::Us => 1_000_000,
        TimeUnit::Ms => 1_000,
        TimeUnit::S => 1,
        TimeUnit::D => vortex_bail!(InvalidArgument: "cannot decode into TimeUnit::D"),
    };

    let days_buf = try_cast(
        array.days(),
        &DType::Primitive(PType::I64, array.dtype().nullability()),
    )?
    .into_primitive()?;
    let mut values: Vec<i64> = days_buf
        .into_maybe_null_slice::<i64>()
        .into_iter()
        .map(|d| d * 86_400 * divisor)
        .collect();

    if let Some(seconds) = array.seconds().as_constant() {
        let seconds =
            PrimitiveScalar::try_from(&seconds.cast(&DType::Primitive(PType::I64, NonNullable))?)?
                .typed_value::<i64>()
                .vortex_expect("non-nullable");
        for v in values.iter_mut() {
            *v += seconds * divisor;
        }
    } else {
        let seconds_buf = try_cast(array.seconds(), &DType::Primitive(PType::U32, NonNullable))?
            .into_primitive()?;
        for (v, second) in values.iter_mut().zip(seconds_buf.maybe_null_slice::<u32>()) {
            *v += (*second as i64) * divisor;
        }
    }

    if let Some(subseconds) = array.subsecond().as_constant() {
        let subseconds = PrimitiveScalar::try_from(
            &subseconds.cast(&DType::Primitive(PType::I64, NonNullable))?,
        )?
        .typed_value::<i64>()
        .vortex_expect("non-nullable");
        for v in values.iter_mut() {
            *v += subseconds;
        }
    } else {
        let subsecond_buf = try_cast(
            array.subsecond(),
            &DType::Primitive(PType::I64, NonNullable),
        )?
        .into_primitive()?;
        for (v, subsecond) in values
            .iter_mut()
            .zip(subsecond_buf.maybe_null_slice::<i64>())
        {
            *v += *subsecond;
        }
    }

    Ok(TemporalArray::new_timestamp(
        PrimitiveArray::from_vec(values, array.validity()).into_array(),
        temporal_metadata.time_unit(),
        temporal_metadata.time_zone().map(ToString::to_string),
    ))
}

#[cfg(test)]
mod test {
    use vortex_array::array::{PrimitiveArray, TemporalArray};
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArrayVariant, ToArrayData};
    use vortex_datetime_dtype::TimeUnit;
    use vortex_dtype::DType;

    use crate::compute::decode_to_temporal;
    use crate::{split_temporal, DateTimePartsArray, TemporalParts};

    #[test]
    fn test_roundtrip_datetimeparts() {
        let raw_values = vec![
            86_400i64,            // element with only day component
            86_400i64 + 1000,     // element with day + second components
            86_400i64 + 1000 + 1, // element with day + second + sub-second components
        ];

        do_roundtrip_test(&raw_values, Validity::NonNullable);
        do_roundtrip_test(&raw_values, Validity::AllValid);
        do_roundtrip_test(&raw_values, Validity::AllInvalid);
        do_roundtrip_test(&raw_values, Validity::from_iter([true, false, true]));
    }

    fn do_roundtrip_test(raw_values: &[i64], validity: Validity) {
        let raw_millis = PrimitiveArray::from_vec(raw_values.to_vec(), validity.clone());
        assert_eq!(raw_millis.validity(), validity);

        let temporal_array = TemporalArray::new_timestamp(
            raw_millis.to_array(),
            TimeUnit::Ms,
            Some("UTC".to_string()),
        );
        assert_eq!(
            temporal_array
                .temporal_values()
                .into_primitive()
                .unwrap()
                .validity(),
            validity
        );

        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(temporal_array.clone()).unwrap();
        assert_eq!(days.clone().into_primitive().unwrap().validity(), validity);
        assert_eq!(
            seconds.clone().into_primitive().unwrap().validity(),
            Validity::NonNullable
        );
        assert_eq!(
            subseconds.clone().into_primitive().unwrap().validity(),
            Validity::NonNullable
        );
        assert_eq!(validity, raw_millis.validity());

        let date_times = DateTimePartsArray::try_new(
            DType::Extension(temporal_array.ext_dtype()),
            days,
            seconds,
            subseconds,
        )
        .unwrap();
        assert_eq!(date_times.validity(), validity);

        let primitive_values = decode_to_temporal(&date_times)
            .unwrap()
            .temporal_values()
            .into_primitive()
            .unwrap();

        assert_eq!(primitive_values.maybe_null_slice::<i64>(), raw_values);
        assert_eq!(primitive_values.validity(), validity);
    }
}
