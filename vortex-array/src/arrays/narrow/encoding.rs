// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Select the stored width of an integer array.
//!
//! Bounds exclude null payloads. Narrowing changes the physical representation while preserving
//! the logical dtype and validity.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::NarrowArray;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::arrays::PrimitiveArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;

impl NarrowArray {
    /// Stores an integer array at the smallest fitting width while retaining its logical dtype.
    ///
    /// Uses the non-null bounds and preserves signedness and validity. Empty and all-null arrays
    /// use the smallest width of the same signedness. Returns the input unchanged if no narrower
    /// type fits. Floating-point arrays are rejected.
    pub fn encode(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let ptype = array.ptype();
        vortex_ensure!(
            ptype.is_int(),
            "Narrow requires integer values, got {ptype}"
        );

        if ptype.byte_width() == 1 {
            return Ok(array.into_array());
        }

        let bounds = min_max(array.as_ref(), ctx, NumericalAggregateOpts::default())?;
        let candidates = if ptype.is_signed_int() {
            [PType::I8, PType::I16, PType::I32]
        } else {
            [PType::U8, PType::U16, PType::U32]
        };

        for candidate in candidates {
            if candidate.byte_width() >= ptype.byte_width() {
                break;
            }

            let storage_dtype = DType::Primitive(candidate, array.dtype().nullability());
            if bounds.as_ref().is_some_and(|bounds| {
                bounds.min.cast(&storage_dtype).is_err() || bounds.max.cast(&storage_dtype).is_err()
            }) {
                continue;
            }

            let values = array
                .as_ref()
                .cast(storage_dtype)?
                .execute::<PrimitiveArray>(ctx)?;

            return Ok(Self::try_new(values.into_array(), array.dtype().clone())?.into_array());
        }

        Ok(array.into_array())
    }
}
