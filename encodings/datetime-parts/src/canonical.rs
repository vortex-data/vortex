// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::{PrimitiveArray, TemporalArray};
use vortex_array::compute::cast;
use vortex_array::validity::Validity;
use vortex_array::vtable::CanonicalVTable;
use vortex_array::{Canonical, IntoArray, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::datetime::{TemporalMetadata, TimeUnit};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect as _, vortex_panic};
use vortex_scalar::PrimitiveScalar;

use crate::{DateTimePartsArray, DateTimePartsVTable};

impl CanonicalVTable<DateTimePartsVTable> for DateTimePartsVTable {
    fn canonicalize(array: &DateTimePartsArray) -> Canonical {
        Canonical::Extension(decode_to_temporal(array).into())
    }
}

/// Decode an [Array] into a [TemporalArray].
///
/// Enforces that the passed array is actually a [DateTimePartsArray] with proper metadata.
pub fn decode_to_temporal(array: &DateTimePartsArray) -> TemporalArray {
    let DType::Extension(ext) = array.dtype().clone() else {
        vortex_panic!(ComputeError: "expected dtype to be DType::Extension variant")
    };

    let Ok(temporal_metadata) = TemporalMetadata::try_from(ext.as_ref()) else {
        vortex_panic!(ComputeError: "must decode TemporalMetadata from extension metadata");
    };

    let divisor = match temporal_metadata.time_unit() {
        TimeUnit::Nanoseconds => 1_000_000_000,
        TimeUnit::Microseconds => 1_000_000,
        TimeUnit::Milliseconds => 1_000,
        TimeUnit::Seconds => 1,
        TimeUnit::Days => vortex_panic!(InvalidArgument: "cannot decode into TimeUnit::D"),
    };

    let days_buf = cast(
        array.days(),
        &DType::Primitive(PType::I64, array.dtype().nullability()),
    )
    .vortex_expect("must be able to cast days to i64")
    .to_primitive();

    // We start with the days component, which is always present.
    // And then add the seconds and subseconds components.
    // We split this into separate passes because often the seconds and/org subseconds components
    // are constant.
    let mut values: BufferMut<i64> = days_buf
        .into_buffer_mut::<i64>()
        .map_each(|d| d * 86_400 * divisor);

    if let Some(seconds) = array.seconds().as_constant() {
        let seconds = PrimitiveScalar::try_from(
            &seconds
                .cast(&DType::Primitive(PType::I64, NonNullable))
                .vortex_expect("must cast to i64"),
        )
        .vortex_expect("must be a primitive")
        .typed_value::<i64>()
        .vortex_expect("non-nullable");
        let seconds = seconds * divisor;
        for v in values.iter_mut() {
            *v += seconds;
        }
    } else {
        let seconds_buf = cast(array.seconds(), &DType::Primitive(PType::U32, NonNullable))
            .vortex_expect("must cast to u32")
            .to_primitive();
        for (v, second) in values.iter_mut().zip(seconds_buf.as_slice::<u32>()) {
            *v += (*second as i64) * divisor;
        }
    }

    if let Some(subseconds) = array.subseconds().as_constant() {
        let subseconds = PrimitiveScalar::try_from(
            &subseconds
                .cast(&DType::Primitive(PType::I64, NonNullable))
                .vortex_expect("must cast to i64"),
        )
        .vortex_expect("must be a primitive")
        .typed_value::<i64>()
        .vortex_expect("non-nullable");
        for v in values.iter_mut() {
            *v += subseconds;
        }
    } else {
        let subsecond_buf = cast(
            array.subseconds(),
            &DType::Primitive(PType::I64, NonNullable),
        )
        .vortex_expect("must cast to i64")
        .to_primitive();
        for (v, subseconds) in values.iter_mut().zip(subsecond_buf.as_slice::<i64>()) {
            *v += *subseconds;
        }
    }

    TemporalArray::new_timestamp(
        PrimitiveArray::new(values.freeze(), Validity::copy_from_array(array.as_ref()))
            .into_array(),
        temporal_metadata.time_unit(),
        temporal_metadata.time_zone().map(ToString::to_string),
    )
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::arrays::{PrimitiveArray, TemporalArray};
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::datetime::TimeUnit;

    use crate::DateTimePartsArray;
    use crate::canonical::decode_to_temporal;

    #[rstest]
    #[case(Validity::NonNullable)]
    #[case(Validity::AllValid)]
    #[case(Validity::AllInvalid)]
    #[case(Validity::from_iter([true, false, true]))]
    fn test_decode_to_temporal(#[case] validity: Validity) {
        let milliseconds = PrimitiveArray::new(
            buffer![
                86_400i64,            // element with only day component
                86_400i64 + 1000,     // element with day + second components
                86_400i64 + 1000 + 1, // element with day + second + sub-second components
            ],
            validity.clone(),
        );
        let date_times = DateTimePartsArray::try_from(TemporalArray::new_timestamp(
            milliseconds.clone().into_array(),
            TimeUnit::Milliseconds,
            Some("UTC".to_string()),
        ))
        .unwrap();

        assert_eq!(
            date_times.validity_mask(),
            validity.to_mask(date_times.len())
        );

        let primitive_values = decode_to_temporal(&date_times)
            .temporal_values()
            .to_primitive();

        assert_eq!(
            primitive_values.as_slice::<i64>(),
            milliseconds.as_slice::<i64>()
        );
        assert_eq!(primitive_values.validity(), &validity);
    }
}
