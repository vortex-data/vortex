// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::DType;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::DateTimeParts;
use crate::array::DateTimePartsArrayExt;
use crate::timestamp;
use crate::timestamp::TimestampParts;

impl OperationsVTable<DateTimeParts> for DateTimeParts {
    fn scalar_at(
        array: ArrayView<'_, DateTimeParts>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let DType::Extension(ext) = array.dtype().clone() else {
            vortex_panic!(
                "DateTimePartsArray must have extension dtype, found {}",
                array.dtype()
            );
        };

        let Some(options) = ext.metadata_opt::<Timestamp>() else {
            vortex_panic!(Compute: "must decode TemporalMetadata from extension metadata");
        };

        if !array.as_ref().is_valid(index, ctx)? {
            return Ok(Scalar::null(DType::Extension(ext)));
        }

        let days: i64 = array
            .days()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("days fits in i64");
        let seconds: i64 = array
            .seconds()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("seconds fits in i64");
        let subseconds: i64 = array
            .subseconds()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<i64>()
            .vortex_expect("subseconds fits in i64");

        let ts = timestamp::combine(
            TimestampParts {
                days,
                seconds,
                subseconds,
            },
            options.unit,
        );

        Ok(Scalar::extension::<Timestamp>(
            options.clone(),
            Scalar::primitive(ts, ext.storage_dtype().nullability()),
        ))
    }
}
