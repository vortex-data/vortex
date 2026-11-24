// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Decimal`] vtable. Its plugin reads and writes the `vortex.decimal` wire form, including
//! legacy nodes.

mod kernel;
mod operations;
mod plugin;
mod validity;

use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayParts;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::decimal::DecimalData;
use crate::arrays::decimal::array::DecimalSlots;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::DecimalBuilder;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::validity::Validity;

/// A [`Decimal`]-encoded Vortex array.
pub type DecimalArray = Array<Decimal>;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

impl ArrayHash for DecimalData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.values.array_hash(state, accuracy);
        std::mem::discriminant(&self.values_type()).hash(state);
    }
}

impl ArrayEq for DecimalData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.values.array_eq(&other.values, accuracy) && self.values_type() == other.values_type()
    }
}

impl VTable for Decimal {
    type TypedArrayData = DecimalData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.decimal");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        fixed_width::buffer("DecimalArray", &array.values, idx)
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        fixed_width::buffer_name(idx)
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        let mut data = array.data().clone();
        data.values = fixed_width::single_buffer(buffers)?;
        Ok(ArrayParts::new(
            self.clone(),
            array.dtype().clone(),
            array.len(),
            data,
            array.slots().iter().cloned().collect(),
        ))
    }

    fn validate(
        &self,
        data: &DecimalData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Decimal(decimal_dtype, nullability) = dtype else {
            vortex_bail!("Expected decimal dtype, got {dtype}");
        };
        vortex_ensure_eq!(
            data.decimal_dtype,
            *decimal_dtype,
            "Decimal values must have the precision and scale of the array",
        );
        DecimalData::validate(&data.values, data.values_type())?;
        vortex_ensure_eq!(
            data.len(),
            len,
            "Decimal values must contain one integer per row",
        );

        let [validity] = slots else {
            vortex_bail!("Decimal requires one validity slot, got {}", slots.len());
        };
        if let Some(child) = validity {
            vortex_ensure!(
                nullability.is_nullable(),
                "A validity child requires a nullable dtype, got {dtype}",
            );
            vortex_ensure_eq!(
                child.dtype(),
                &Validity::DTYPE,
                "Decimal validity must contain non-nullable booleans",
            );
            vortex_ensure_eq!(
                child.len(),
                len,
                "Decimal validity must contain one boolean per row",
            );
        }

        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        DecimalSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some(builder) = builder.as_any_mut().downcast_mut::<DecimalBuilder>() else {
            vortex_bail!("append_to_builder for Decimal requires a DecimalBuilder");
        };
        builder.append_decimal_array(array, ctx)
    }
}

/// The vtable of [`DecimalArray`].
#[derive(Clone, Debug)]
pub struct Decimal;

#[cfg(test)]
mod tests;
