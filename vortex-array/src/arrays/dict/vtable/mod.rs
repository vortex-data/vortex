// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use kernel::PARENT_KERNELS;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use super::DictArrayParts;
use super::DictData;
use super::DictMetadata;
use super::array::NUM_SLOTS;
use super::array::SLOT_NAMES;
use super::take_canonical;
use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::DeserializeMetadata;
use crate::Precision;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Primitive;
use crate::arrays::constant::ConstantData;
use crate::arrays::dict::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::require_child;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::vtable;
mod kernel;
mod operations;
mod validity;

vtable!(Dict, Dict, DictData);

#[derive(Clone, Debug)]
pub struct Dict;

impl Dict {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.dict");
}

impl VTable for Dict {
    type ArrayData = DictData;

    type Metadata = ProstMetadata<DictMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(_array: &Self::ArrayData) -> &Self {
        &Dict
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &DictData) -> usize {
        array.codes().len()
    }

    fn dtype(array: &DictData) -> &DType {
        &array.dtype
    }

    fn stats(array: &DictData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(array: &DictData, state: &mut H, precision: Precision) {
        array.codes().array_hash(state, precision);
        array.values().array_hash(state, precision);
    }

    fn array_eq(array: &DictData, other: &DictData, precision: Precision) -> bool {
        array.codes().array_eq(other.codes(), precision)
            && array.values().array_eq(other.values(), precision)
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

    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DictMetadata {
            codes_ptype: PType::try_from(array.codes().dtype())? as i32,
            values_len: u32::try_from(array.values().len()).map_err(|_| {
                vortex_err!(
                    "Dictionary values size {} overflowed u32",
                    array.values().len()
                )
            })?,
            is_nullable_codes: Some(array.codes().dtype().is_nullable()),
            all_values_referenced: Some(array.all_values_referenced),
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let metadata = <Self::Metadata as DeserializeMetadata>::deserialize(bytes)?;
        Ok(ProstMetadata(metadata))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DictData> {
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

        // SAFETY: We've validated the metadata and children.
        Ok(unsafe {
            DictData::new_unchecked(codes, values).set_all_values_referenced(all_values_referenced)
        })
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "DictArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        if array.is_empty() {
            let result_dtype = array
                .dtype()
                .union_nullability(array.codes().dtype().nullability());
            return Ok(ExecutionResult::done(Canonical::empty(&result_dtype)));
        }

        let array = require_child!(array, array.codes(), 0 => Primitive);

        // TODO(joe): use stat get instead computing.
        // Also not the check to do here it take value validity using code validity, but this approx
        // is correct.
        if array.codes().all_invalid()? {
            return Ok(ExecutionResult::done(ConstantData::new(
                Scalar::null(array.dtype().as_nullable()),
                array.codes().len(),
            )));
        }

        let array = require_child!(array, array.values(), 1 => AnyCanonical);

        let DictArrayParts { codes, values, .. } = array.into_data().into_parts();

        let codes = codes
            .try_into::<Primitive>()
            .ok()
            .vortex_expect("must be primitive");
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
