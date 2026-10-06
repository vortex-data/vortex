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

        let dtype = array.dtype().clone();
        let values = Self::encode_values(array, ctx)?;
        if values.dtype() == &dtype {
            return Ok(values.into_array());
        }

        Ok(Self::try_new(values.into_array(), dtype)?.into_array())
    }

    /// Selects the stored primitive width for internal buffers whose dtype is chosen at construction.
    ///
    /// This changes the returned dtype while preserving signedness and validity. For arrays with
    /// an existing logical dtype contract, use [`Self::encode`] to retain that dtype. Floating-point
    /// arrays are returned unchanged. Empty and all-null integers use the smallest width of the
    /// same signedness.
    pub fn encode_values(
        array: PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        let ptype = array.ptype();
        if !ptype.is_int() || ptype.byte_width() == 1 {
            return Ok(array);
        }
        let bounds = min_max(array.as_ref(), ctx, NumericalAggregateOpts::default())?;
        let storage_type = if let Some(bounds) = bounds {
            let smallest = if ptype.is_signed_int() {
                (
                    PType::min_signed_ptype_for_value(i64::try_from(&bounds.min)?),
                    PType::min_signed_ptype_for_value(i64::try_from(&bounds.max)?),
                )
            } else {
                (
                    PType::min_unsigned_ptype_for_value(u64::try_from(&bounds.min)?),
                    PType::min_unsigned_ptype_for_value(u64::try_from(&bounds.max)?),
                )
            };
            if smallest.0.byte_width() >= smallest.1.byte_width() {
                smallest.0
            } else {
                smallest.1
            }
        } else if ptype.is_signed_int() {
            PType::I8
        } else {
            PType::U8
        };
        if storage_type.byte_width() >= ptype.byte_width() {
            return Ok(array);
        }

        array
            .as_ref()
            .cast(DType::Primitive(storage_type, array.dtype().nullability()))?
            .execute(ctx)
    }
}

/// Integer widths in increasing order with the input's signedness.
pub(super) fn integer_types(ptype: PType) -> [PType; 4] {
    if ptype.is_signed_int() {
        [PType::I8, PType::I16, PType::I32, PType::I64]
    } else {
        [PType::U8, PType::U16, PType::U32, PType::U64]
    }
}
