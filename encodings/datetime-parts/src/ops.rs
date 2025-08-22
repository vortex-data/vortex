// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_dtype::DType;
use vortex_dtype::datetime::TemporalMetadata;
use vortex_error::{VortexExpect, vortex_panic};
use vortex_scalar::Scalar;

use crate::timestamp::TimestampParts;
use crate::{DateTimePartsArray, DateTimePartsVTable, timestamp};

impl OperationsVTable<DateTimePartsVTable> for DateTimePartsVTable {
    fn slice(array: &DateTimePartsArray, start: usize, stop: usize) -> ArrayRef {
        // SAFETY: slicing all components preserves values
        unsafe {
            DateTimePartsArray::new_unchecked(
                array.dtype().clone(),
                array.days().slice(start, stop),
                array.seconds().slice(start, stop),
                array.subseconds().slice(start, stop),
            )
            .into_array()
        }
    }

    fn scalar_at(array: &DateTimePartsArray, index: usize) -> Scalar {
        let DType::Extension(ext) = array.dtype().clone() else {
            vortex_panic!(
                "DateTimePartsArray must have extension dtype, found {}",
                array.dtype()
            );
        };

        let Ok(temporal_metadata) = TemporalMetadata::try_from(ext.as_ref()) else {
            vortex_panic!(ComputeError: "must decode TemporalMetadata from extension metadata");
        };

        if !array.is_valid(index).vortex_expect("validity access") {
            return Scalar::null(DType::Extension(ext));
        }

        let days: i64 = array
            .days()
            .scalar_at(index)
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("days fits in i64");
        let seconds: i64 = array
            .seconds()
            .scalar_at(index)
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("seconds fits in i64");
        let subseconds: i64 = array
            .subseconds()
            .scalar_at(index)
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("subseconds fits in i64");

        let ts = timestamp::combine(
            TimestampParts {
                days,
                seconds,
                subseconds,
            },
            temporal_metadata.time_unit(),
        );

        Scalar::extension(ext, Scalar::from(ts))
    }
}
