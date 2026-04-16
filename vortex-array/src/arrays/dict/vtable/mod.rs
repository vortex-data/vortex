// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use kernel::PARENT_KERNELS;
use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::DictData;
use super::DictMetadata;
use super::array::DictSlots;
use super::array::DictSlotsView;
use super::take_canonical;
use crate::AnyCanonical;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::ConstantArray;
use crate::arrays::Primitive;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::dict::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::require_child;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
mod kernel;
mod operations;
mod validity;

/// A [`Dict`]-encoded Vortex array.
pub type DictArray = Array<Dict>;

#[derive(Clone, Debug)]
pub struct Dict;

impl ArrayHash for DictData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _precision: Precision) {}
}

impl ArrayEq for DictData {
    fn array_eq(&self, _other: &Self, _precision: Precision) -> bool {
        true
    }
}

impl VTable for Dict {
    type ArrayData = DictData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.dict");
        *ID
    }

    fn validate(
        &self,
        _data: &DictData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let view = DictSlotsView::from_slots(slots);
        let codes = view.codes;
        let values = view.values;
        vortex_ensure!(codes.len() == len, "DictArray codes length mismatch");
        vortex_ensure!(
            values
                .dtype()
                .union_nullability(codes.dtype().nullability())
                == *dtype,
            "DictArray dtype does not match codes/values dtype"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("DictArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            DictMetadata {
                codes_ptype: PType::try_from(array.codes().dtype())? as i32,
                values_len: u32::try_from(array.values().len()).map_err(|_| {
                    vortex_err!(
                        "Dictionary values size {} overflowed u32",
                        array.values().len()
                    )
                })?,
                is_nullable_codes: Some(array.codes().dtype().is_nullable()),
                all_values_referenced: Some(array.has_all_values_referenced()),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],

        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<crate::array::ArrayParts<Self>> {
        let metadata = DictMetadata::decode(metadata)?;
        if children.len() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                children.len()
            )
        }
        let codes_nullable = metadata
            .is_nullable_codes
            .map(Nullability::from)
            // If no `is_nullable_codes` metadata use the nullability of the values
            // (and whole array) as before.
            .unwrap_or_else(|| dtype.nullability());
        let codes_dtype = DType::Primitive(metadata.codes_ptype(), codes_nullable);
        let codes = children.get(0, &codes_dtype, len)?;
        let values = children.get(1, dtype, metadata.values_len as usize)?;
        let all_values_referenced = metadata.all_values_referenced.unwrap_or(false);

        Ok(
            crate::array::ArrayParts::new(self.clone(), dtype.clone(), len, unsafe {
                DictData::new_unchecked().set_all_values_referenced(all_values_referenced)
            })
            .with_slots(vec![Some(codes), Some(values)]),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        DictSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        if array.is_empty() {
            let result_dtype = array
                .dtype()
                .union_nullability(array.codes().dtype().nullability());
            return Ok(ExecutionResult::done(Canonical::empty(&result_dtype)));
        }

        let array = require_child!(array, array.codes(), DictSlots::CODES => Primitive);

        // TODO(joe): use stat get instead computing.
        // Also not the check to do here it take value validity using code validity, but this approx
        // is correct.
        if array.codes().all_invalid(ctx)? {
            return Ok(ExecutionResult::done(ConstantArray::new(
                Scalar::null(array.dtype().as_nullable()),
                array.codes().len(),
            )));
        }

        let array = require_child!(array, array.values(), DictSlots::VALUES => AnyCanonical);

        let codes = array
            .codes()
            .clone()
            .try_downcast::<Primitive>()
            .ok()
            .vortex_expect("must be primitive");
        let values = array.values().clone();
        debug_assert!(values.is_canonical());
        // TODO: add canonical owned cast.
        let values = values.to_canonical()?;

        Ok(ExecutionResult::done(take_canonical(values, &codes, ctx)?))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}
