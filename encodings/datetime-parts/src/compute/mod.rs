mod cast;
mod filter;
mod take;

use vortex_array::compute::{scalar_at, slice, CastFn, FilterFn, ScalarAtFn, SliceFn, TakeFn};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{ArrayData, IntoArrayData};
use vortex_datetime_dtype::{TemporalMetadata, TimeUnit};
use vortex_dtype::Nullability::{NonNullable, Nullable};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexResult};
use vortex_scalar::Scalar;

use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl ComputeVTable for DateTimePartsEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<ArrayData>> {
        Some(self)
    }

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

        if !array.is_valid(index)? {
            return Ok(Scalar::null(DType::Extension(ext)));
        }

        let divisor = match time_unit {
            TimeUnit::Ns => 1_000_000_000,
            TimeUnit::Us => 1_000_000,
            TimeUnit::Ms => 1_000,
            TimeUnit::S => 1,
            TimeUnit::D => vortex_bail!("Invalid time unit D"),
        };

        let days: i64 = scalar_at(array.days(), index)?
            .cast(&DType::Primitive(PType::I64, Nullable))?
            .try_into()?;
        let seconds: i64 = scalar_at(array.seconds(), index)?
            .cast(&DType::Primitive(PType::I64, NonNullable))?
            .try_into()?;
        let subseconds: i64 = scalar_at(array.subsecond(), index)?
            .cast(&DType::Primitive(PType::I64, NonNullable))?
            .try_into()?;

        let scalar = days * 86_400 * divisor + seconds * divisor + subseconds;

        Ok(Scalar::extension(ext, Scalar::from(scalar)))
    }
}
