use itertools::Itertools as _;
use vortex_array::array::{PrimitiveArray, TemporalArray};
use vortex_array::compute::unary::{scalar_at, ScalarAtFn};
use vortex_array::compute::{slice, take, ArrayCompute, SliceFn, TakeFn};
use vortex_array::validity::ArrayValidity;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_datetime_dtype::{TemporalMetadata, TimeUnit};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult, VortexUnwrap as _};
use vortex_scalar::{Scalar, ScalarValue};

use crate::DateTimePartsArray;

impl ArrayCompute for DateTimePartsArray {
    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }
}

impl TakeFn for DateTimePartsArray {
    fn take(&self, indices: &ArrayData) -> VortexResult<ArrayData> {
        Ok(Self::try_new(
            self.dtype().clone(),
            take(self.days(), indices)?,
            take(self.seconds(), indices)?,
            take(self.subsecond(), indices)?,
        )?
        .into_array())
    }
}

impl SliceFn for DateTimePartsArray {
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(Self::try_new(
            self.dtype().clone(),
            slice(self.days(), start, stop)?,
            slice(self.seconds(), start, stop)?,
            slice(self.subsecond(), start, stop)?,
        )?
        .into_array())
    }
}

impl ScalarAtFn for DateTimePartsArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let DType::Extension(ext) = self.dtype().clone() else {
            vortex_bail!(
                "DateTimePartsArray must have extension dtype, found {}",
                self.dtype()
            );
        };

        let TemporalMetadata::Timestamp(time_unit, _) = TemporalMetadata::try_from(ext.as_ref())?
        else {
            vortex_bail!("Metadata must be Timestamp, found {}", ext.id());
        };

        if !self.is_valid(index) {
            return Ok(Scalar::extension(ext, ScalarValue::Null));
        }

        let divisor = match time_unit {
            TimeUnit::Ns => 1_000_000_000,
            TimeUnit::Us => 1_000_000,
            TimeUnit::Ms => 1_000,
            TimeUnit::S => 1,
            TimeUnit::D => vortex_bail!("Invalid time unit D"),
        };

        let days: i64 = scalar_at(self.days(), index)?.try_into()?;
        let seconds: i64 = scalar_at(self.seconds(), index)?.try_into()?;
        let subseconds: i64 = scalar_at(self.subsecond(), index)?.try_into()?;

        let scalar = days * 86_400 * divisor + seconds * divisor + subseconds;

        Ok(Scalar::extension(ext, scalar.into()))
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        <Self as ScalarAtFn>::scalar_at(self, index).vortex_unwrap()
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

    let days_buf = array.days().into_primitive()?;
    let seconds_buf = array.seconds().into_primitive()?;
    let subsecond_buf = array.subsecond().into_primitive()?;

    let values = days_buf
        .maybe_null_slice::<i64>()
        .iter()
        .zip_eq(seconds_buf.maybe_null_slice::<i64>().iter())
        .zip_eq(subsecond_buf.maybe_null_slice::<i64>().iter())
        .map(|((d, s), ss)| d * 86_400 * divisor + s * divisor + ss)
        .collect::<Vec<_>>();

    Ok(TemporalArray::new_timestamp(
        PrimitiveArray::from_vec(values, array.validity().clone()).into_array(),
        temporal_metadata.time_unit(),
        temporal_metadata.time_zone().map(ToString::to_string),
    ))
}

#[cfg(test)]
mod test {
    use vortex_array::array::{PrimitiveArray, TemporalArray};
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArrayData, IntoArrayVariant};
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
        do_roundtrip_test(&raw_values, Validity::from(vec![true, false, true]));
    }

    fn do_roundtrip_test(raw_values: &[i64], validity: Validity) {
        let raw_millis = PrimitiveArray::from_vec(raw_values.to_vec(), validity.clone());
        assert_eq!(raw_millis.validity(), validity);

        let temporal_array = TemporalArray::new_timestamp(
            raw_millis.clone().into_array(),
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
        assert_eq!(days.as_primitive().validity(), validity);
        assert_eq!(seconds.as_primitive().validity(), Validity::NonNullable);
        assert_eq!(subseconds.as_primitive().validity(), Validity::NonNullable);
        assert_eq!(validity, raw_millis.validity());

        let date_times = DateTimePartsArray::try_new(
            DType::Extension(temporal_array.ext_dtype().clone()),
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
