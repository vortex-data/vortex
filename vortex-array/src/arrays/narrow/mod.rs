// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer arrays that retain their logical dtype while storing a narrower integer child.
//!
//! [`NarrowArray`] stores no buffers of its own and derives validity from its child. Selection,
//! comparison, and aggregate operations can use that child without widening to the logical dtype.
//! Canonical execution widens to the logical integer dtype.

mod aggregates;
pub(crate) use aggregates::register_aggregate_kernels;
mod compare;
mod encoding;
mod rules;
mod vtable;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::array_slots;
use crate::dtype::DType;
use crate::dtype::integer::integer_byte_width;
use crate::dtype::integer::signed_integer_type;

/// An integer encoding with a narrower child of the same signedness.
#[derive(Clone, Debug)]
pub struct Narrow;

/// An integer array whose logical dtype is wider than its stored values.
pub type NarrowArray = Array<Narrow>;

/// The child of a [`NarrowArray`].
#[array_slots(Narrow)]
pub struct NarrowSlots {
    /// Integer values with the same signedness and nullability as the logical dtype.
    #[slot(0)]
    pub values: ArrayRef,
}

impl NarrowArray {
    /// Wraps an integer child with a wider logical dtype without reading its values.
    ///
    /// Both dtypes must be integers with the same signedness and nullability. The child must be
    /// strictly narrower than `dtype` and may use any encoding. Nested Narrow arrays are flattened.
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
}

pub(super) fn validate_dtypes(storage: &DType, logical: &DType) -> VortexResult<()> {
    let storage_width = integer_byte_width(storage)
        .ok_or_else(|| vortex_error::vortex_err!("Expected integer storage, got {storage}"))?;
    let logical_width = integer_byte_width(logical)
        .ok_or_else(|| vortex_error::vortex_err!("Expected integer dtype, got {logical}"))?;
    vortex_ensure!(
        signed_integer_type(storage).is_some() == signed_integer_type(logical).is_some(),
        "Narrow requires matching signedness, got {storage} and {logical}"
    );
    vortex_ensure!(
        storage_width < logical_width,
        "Narrow requires storage narrower than {logical}, got {storage}"
    );
    vortex_ensure!(
        storage.nullability() == logical.nullability(),
        "Narrow requires matching nullability, got {storage} and {logical}"
    );

    Ok(())
}

pub(crate) fn initialize(session: &VortexSession) {
    compare::initialize(session);
    rules::initialize(session);
}

#[cfg(test)]
mod tests;
