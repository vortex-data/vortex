mod cast;
mod compare;
mod filter;
mod take;

use vortex_array::compute::{
    CastFn, CompareFn, FilterFn, ScalarAtFn, SliceFn, TakeFn, scalar_at, slice,
};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::Nullability::{NonNullable, Nullable};
use vortex_dtype::datetime::TemporalMetadata;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::timestamp::{self, TimestampParts};
use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl ComputeVTable for DateTimePartsEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<&dyn Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<&dyn Array>> {
        Some(self)
    }

    // TODO(joe): implement `between_fn` this is used at lot.
}

impl SliceFn<&DateTimePartsArray> for DateTimePartsEncoding {
    fn slice(
        &self,
        array: &DateTimePartsArray,
        start: usize,
        stop: usize,
    ) -> VortexResult<ArrayRef> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            slice(array.days(), start, stop)?,
            slice(array.seconds(), start, stop)?,
            slice(array.subseconds(), start, stop)?,
        )?
        .into_array())
    }
}

impl ScalarAtFn<&DateTimePartsArray> for DateTimePartsEncoding {
    fn scalar_at(&self, array: &DateTimePartsArray, index: usize) -> VortexResult<Scalar> {
        let DType::Extension(ext) = array.dtype().clone() else {
            vortex_bail!(
                "DateTimePartsArray must have extension dtype, found {}",
                array.dtype()
            );
        };

        let Ok(temporal_metadata) = TemporalMetadata::try_from(ext.as_ref()) else {
            vortex_bail!(ComputeError: "must decode TemporalMetadata from extension metadata");
        };

        if !array.is_valid(index)? {
            return Ok(Scalar::null(DType::Extension(ext)));
        }

        let days: i64 = scalar_at(array.days(), index)?
            .cast(&DType::Primitive(PType::I64, Nullable))?
            .try_into()?;
        let seconds: i64 = scalar_at(array.seconds(), index)?
            .cast(&DType::Primitive(PType::I64, NonNullable))?
            .try_into()?;
        let subseconds: i64 = scalar_at(array.subseconds(), index)?
            .cast(&DType::Primitive(PType::I64, NonNullable))?
            .try_into()?;

        let ts = timestamp::combine(
            TimestampParts {
                days,
                seconds,
                subseconds,
            },
            temporal_metadata.time_unit(),
        )?;

        Ok(Scalar::extension(ext, Scalar::from(ts)))
    }
}
