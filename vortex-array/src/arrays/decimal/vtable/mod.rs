// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal vtable and adapters for the existing `vortex.decimal` wire form.
//!
//! Stored-node validation is shared by the canonical reader and the compatibility plugin, so both
//! readers reject malformed buffer layouts before constructing arrays.

mod kernel;
mod operations;

mod plugin;
pub use plugin::DecimalPlugin;

mod stored;
use stored::StoredDecimal;

mod validity;

use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
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
use crate::arrays::decimal::compute::rules::RULES;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::DecimalBuilder;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::serde::ArrayChildren;

/// A [`Decimal`]-encoded Vortex array.
pub type DecimalArray = Array<Decimal>;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

/// The stored integer width of a `vortex.decimal` node.
#[derive(prost::Message)]
pub struct DecimalMetadata {
    /// The [`DecimalType`] discriminant stored in the wire metadata.
    #[prost(int32, tag = "1")]
    pub(super) values_type: i32,
}

impl DecimalMetadata {
    fn get_values_type(&self) -> VortexResult<DecimalType> {
        DecimalType::try_from(self.values_type).map_err(|_| {
            vortex_err!(
                "DecimalMetadata values_type must be a known DecimalType, got {}",
                self.values_type
            )
        })
    }
}

impl ArrayHash for DecimalData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.values.array_hash(state, accuracy);
        std::mem::discriminant(&self.values_type).hash(state);
    }
}

impl ArrayEq for DecimalData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.values.array_eq(&other.values, accuracy) && self.values_type == other.values_type
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

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            DecimalMetadata {
                values_type: array.values_type() as i32,
            }
            .encode_to_vec(),
        ))
    }

    fn validate(
        &self,
        data: &DecimalData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Decimal(_, nullability) = dtype else {
            vortex_bail!("Expected decimal dtype, got {dtype:?}");
        };
        vortex_ensure_eq!(
            data.len(),
            len,
            "DecimalArray length does not match outer length",
        );
        let validity =
            crate::array::child_to_validity(slots[DecimalSlots::VALIDITY].as_ref(), *nullability);
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure_eq!(
                validity_len,
                len,
                "DecimalArray validity len does not match outer length",
            );
        }

        Ok(())
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let stored = StoredDecimal::deserialize(dtype, len, metadata, buffers, children)?;
        let canonical_type = stored.decimal_dtype.canonical_values_type();
        vortex_ensure!(
            stored.values_type == canonical_type,
            "Decimal reads only the canonical width {canonical_type} of {}, and DecimalPlugin \
             reads every other width, got {}",
            stored.decimal_dtype,
            stored.values_type,
        );

        stored.into_array_parts()
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
        builder.append_decimal_array(&array.into_owned(), ctx)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

#[derive(Clone, Debug)]
pub struct Decimal;

#[cfg(test)]
mod tests;
