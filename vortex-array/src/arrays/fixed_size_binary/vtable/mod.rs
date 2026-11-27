// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`FixedSizeBinary`] vtable.

mod kernel;
mod operations;
mod plugin;
mod validity;

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
use crate::arrays::fixed_size_binary::FixedSizeBinaryData;
use crate::arrays::fixed_size_binary::FixedSizeBinarySlots;
use crate::arrays::fixed_size_binary::array::values_len;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::FixedSizeBinaryBuilder;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::validity::Validity;

/// A [`FixedSizeBinary`]-encoded Vortex array.
pub type FixedSizeBinaryArray = Array<FixedSizeBinary>;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

impl ArrayHash for FixedSizeBinaryData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.values.array_hash(state, accuracy);
    }
}

impl ArrayEq for FixedSizeBinaryData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.values.array_eq(&other.values, accuracy)
    }
}

impl VTable for FixedSizeBinary {
    type TypedArrayData = FixedSizeBinaryData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.fixed_size_binary");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        fixed_width::buffer("FixedSizeBinaryArray", &array.values, idx)
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
        data: &FixedSizeBinaryData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::FixedSizeBinary(byte_width, nullability) = dtype else {
            vortex_bail!("Expected fixed-size binary dtype, got {dtype}");
        };
        vortex_ensure_eq!(
            data.values.len(),
            values_len(len, *byte_width)?,
            "FixedSizeBinary values must contain {byte_width} bytes per row",
        );

        let [validity] = slots else {
            vortex_bail!(
                "FixedSizeBinary requires one validity slot, got {}",
                slots.len()
            );
        };
        if let Some(child) = validity {
            vortex_ensure!(
                nullability.is_nullable(),
                "A validity child requires a nullable dtype, got {dtype}",
            );
            vortex_ensure_eq!(
                child.dtype(),
                &Validity::DTYPE,
                "FixedSizeBinary validity must contain non-nullable booleans",
            );
            vortex_ensure_eq!(
                child.len(),
                len,
                "FixedSizeBinary validity must contain one boolean per row",
            );
        }

        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        FixedSizeBinarySlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some(builder) = builder
            .as_any_mut()
            .downcast_mut::<FixedSizeBinaryBuilder>()
        else {
            vortex_bail!("append_to_builder for FixedSizeBinary requires a FixedSizeBinaryBuilder");
        };
        builder.append_fixed_size_binary_array(array, ctx)
    }
}

/// The vtable of [`FixedSizeBinaryArray`].
#[derive(Clone, Debug)]
pub struct FixedSizeBinary;

#[cfg(test)]
mod tests;
