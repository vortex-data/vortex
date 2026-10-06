// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Select the stored width of an integer array.
//!
//! Bounds exclude null payloads. Narrowing changes the physical representation while preserving
//! the logical dtype and validity.

use vortex_error::VortexExpect;
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
use crate::dtype::DecimalType;
use crate::dtype::PType;
use crate::dtype::integer::integer_dtype;
use crate::dtype::integer::signed_integer_type;
use crate::integer;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;

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
            let (min_type, max_type) = if ptype.is_signed_int() {
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
            if min_type.byte_width() >= max_type.byte_width() {
                min_type
            } else {
                max_type
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

    /// Narrows signed primitive or wide integer values without changing their logical dtype.
    pub fn encode_signed(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let logical = signed_integer_type(array.dtype());
        vortex_ensure!(
            logical.is_some(),
            "Expected signed integers, got {}",
            array.dtype()
        );
        let logical = logical.vortex_expect("Signed integer dtype checked above");
        let storage = integer::storage_child(&array);
        let bounds = min_max(storage, ctx, NumericalAggregateOpts::default())?
            .map(|bounds| {
                Ok::<_, vortex_error::VortexError>((
                    integer::scalar_value(&bounds.min)?.as_i256(),
                    integer::scalar_value(&bounds.max)?.as_i256(),
                ))
            })
            .transpose()?;
        for candidate in [
            DecimalType::I8,
            DecimalType::I16,
            DecimalType::I32,
            DecimalType::I64,
            DecimalType::I128,
        ] {
            if candidate >= logical {
                break;
            }
            let fits = match_each_decimal_value_type!(candidate, |T| {
                bounds.is_none_or(|(min, max)| {
                    DecimalValue::I256(min).cast::<T>().is_some()
                        && DecimalValue::I256(max).cast::<T>().is_some()
                })
            });
            if fits {
                let dtype = integer_dtype(candidate, array.dtype().nullability());
                let values = integer::cast_array(storage, &dtype, ctx)?;
                return Ok(Self::try_new(values, array.dtype().clone())?.into_array());
            }
        }
        Ok(array)
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
