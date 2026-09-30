// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer arrays backed by a narrower integer child of the same signedness.
//!
//! [`NarrowArray`] preserves the logical dtype while selection and comparison kernels operate
//! on its child. Canonical execution widens the values into a [`PrimitiveArray`].

mod compute;
mod vtable;

#[cfg(test)]
mod tests;

pub(crate) use compute::initialize;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::Array;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::array_slots;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;

/// An integer encoding whose child has a smaller width and the same signedness.
#[derive(Clone, Debug)]
pub struct Narrow;

/// A logically wide integer array backed by narrower integer values.
pub type NarrowArray = Array<Narrow>;

/// The physical child of a [`NarrowArray`].
#[array_slots(Narrow)]
pub struct NarrowSlots {
    /// Values with the same signedness and nullability as the logical dtype.
    #[slot(0)]
    pub values: ArrayRef,
}

impl NarrowArray {
    /// Wrap narrower integer values with a wider logical dtype, without reading their buffers.
    ///
    /// The dtypes must have the same signedness and nullability, and `dtype` must be strictly
    /// wider. The child may use any encoding. Nested narrow wrappers are flattened.
    pub fn try_new(mut values: ArrayRef, dtype: DType) -> VortexResult<Self> {
        validate_dtypes(values.dtype(), &dtype)?;
        while let Some(inner) = values.as_opt::<Narrow>() {
            values = inner.values().clone();
        }
        let len = values.len();
        Self::try_from_parts(
            ArrayParts::new(Narrow, dtype, len, EmptyArrayData)
                .with_slots(NarrowSlots { values }.into_slots()),
        )
    }

    /// Narrow an integer buffer while preserving its logical dtype and signedness.
    ///
    /// Selects storage using the non-null minimum and maximum, then casts the values once.
    /// Returns the original array when no smaller type in the same signedness family fits.
    pub fn encode(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let ptype = array.ptype();
        vortex_ensure!(ptype.is_int(), "NarrowArray requires integer values");
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

pub(super) fn validate_dtypes(storage: &DType, logical: &DType) -> VortexResult<()> {
    let storage_ptype = PType::try_from(storage)?;
    let logical_ptype = PType::try_from(logical)?;
    vortex_ensure!(
        storage_ptype.is_int() && logical_ptype.is_int(),
        "NarrowArray requires integer dtypes, got {storage} and {logical}"
    );
    vortex_ensure!(
        storage_ptype.is_signed_int() == logical_ptype.is_signed_int(),
        "NarrowArray must preserve signedness, got {storage} and {logical}"
    );
    vortex_ensure!(
        storage_ptype.byte_width() < logical_ptype.byte_width(),
        "NarrowArray storage {storage} must be strictly narrower than {logical}"
    );
    vortex_ensure!(
        storage.nullability() == logical.nullability(),
        "NarrowArray must preserve nullability, got {storage} and {logical}"
    );
    Ok(())
}
