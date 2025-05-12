use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_dtype::Nullability::{NonNullable, Nullable};
use vortex_dtype::datetime::TemporalMetadata;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::timestamp::TimestampParts;
use crate::{DateTimePartsArray, DateTimePartsVTable, timestamp};

impl OperationsVTable<DateTimePartsVTable> for DateTimePartsVTable {
    fn slice(array: &DateTimePartsArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(DateTimePartsArray::try_new(
            array.dtype().clone(),
            array.days().slice(start, stop)?,
            array.seconds().slice(start, stop)?,
            array.subseconds().slice(start, stop)?,
        )?
        .into_array())
    }

    fn scalar_at(array: &DateTimePartsArray, index: usize) -> VortexResult<Scalar> {
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

        let days: i64 = array
            .days()
            .scalar_at(index)?
            .cast(&DType::Primitive(PType::I64, Nullable))?
            .try_into()?;
        let seconds: i64 = array
            .seconds()
            .scalar_at(index)?
            .cast(&DType::Primitive(PType::I64, NonNullable))?
            .try_into()?;
        let subseconds: i64 = array
            .subseconds()
            .scalar_at(index)?
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
