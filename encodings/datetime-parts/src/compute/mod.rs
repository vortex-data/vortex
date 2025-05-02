mod cast;
mod compare;
mod filter;
mod is_constant;
mod take;

use vortex_array::Array;
use vortex_array::compute::{ScalarAtFn, TakeFn, scalar_at};
use vortex_array::vtable::ComputeVTable;
use vortex_dtype::Nullability::{NonNullable, Nullable};
use vortex_dtype::datetime::TemporalMetadata;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::timestamp::{self, TimestampParts};
use crate::{DateTimePartsArray, DateTimePartsEncoding};

impl ComputeVTable for DateTimePartsEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    // TODO(joe): implement `between_fn` this is used at lot.
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
