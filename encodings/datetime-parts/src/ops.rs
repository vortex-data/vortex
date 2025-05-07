use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_dtype::Nullability::{NonNullable, Nullable};
use vortex_dtype::datetime::TemporalMetadata;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::timestamp::TimestampParts;
use crate::{DateTimePartsArray, timestamp};

impl ArrayOperationsImpl for DateTimePartsArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(DateTimePartsArray::try_new(
            self.dtype().clone(),
            self.days().slice(start, stop)?,
            self.seconds().slice(start, stop)?,
            self.subseconds().slice(start, stop)?,
        )?
        .into_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let DType::Extension(ext) = self.dtype().clone() else {
            vortex_bail!(
                "DateTimePartsArray must have extension dtype, found {}",
                self.dtype()
            );
        };

        let Ok(temporal_metadata) = TemporalMetadata::try_from(ext.as_ref()) else {
            vortex_bail!(ComputeError: "must decode TemporalMetadata from extension metadata");
        };

        if !self.is_valid(index)? {
            return Ok(Scalar::null(DType::Extension(ext)));
        }

        let days: i64 = self
            .days()
            .scalar_at(index)?
            .cast(&DType::Primitive(PType::I64, Nullable))?
            .try_into()?;
        let seconds: i64 = self
            .seconds()
            .scalar_at(index)?
            .cast(&DType::Primitive(PType::I64, NonNullable))?
            .try_into()?;
        let subseconds: i64 = self
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
