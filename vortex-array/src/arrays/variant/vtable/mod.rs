// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operations;
mod validity;

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_proto::dtype as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::VTable;
use crate::arrays::variant::CORE_STORAGE_SLOT;
use crate::arrays::variant::NUM_SLOTS;
use crate::arrays::variant::SHREDDED_SLOT;
use crate::arrays::variant::SLOT_NAMES;
use crate::arrays::variant::compute::rules::RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;

/// A [`Variant`]-encoded Vortex array.
pub type VariantArray = Array<Variant>;

#[derive(Clone, Debug)]
pub struct Variant;

#[derive(Clone, prost::Message)]
struct VariantMetadataProto {
    #[prost(message, optional, tag = "1")]
    pub shredded_dtype: Option<pb::DType>,
}

impl VTable for Variant {
    type TypedArrayData = EmptyArrayData;

    type OperationsVTable = Self;

    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.variant");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "VariantArray expects {NUM_SLOTS} slots, got {}",
            slots.len()
        );
        vortex_ensure!(
            slots[CORE_STORAGE_SLOT].is_some(),
            "VariantArray core_storage slot must be present"
        );
        let core_storage = slots[CORE_STORAGE_SLOT]
            .as_ref()
            .vortex_expect("validated core_storage slot presence");
        vortex_ensure!(
            matches!(dtype, DType::Variant(_)),
            "Expected Variant DType, got {dtype}"
        );
        vortex_ensure!(
            matches!(core_storage.dtype(), DType::Variant(_)),
            "VariantArray core_storage dtype must be Variant, found {}",
            core_storage.dtype()
        );
        vortex_ensure!(
            core_storage.dtype() == dtype,
            "VariantArray core_storage dtype {} does not match outer dtype {}",
            core_storage.dtype(),
            dtype
        );
        vortex_ensure!(
            core_storage.len() == len,
            "VariantArray core_storage length {} does not match outer length {}",
            core_storage.len(),
            len
        );
        if let Some(shredded) = slots[SHREDDED_SLOT].as_ref() {
            vortex_ensure!(
                shredded.len() == len,
                "VariantArray shredded length {} does not match outer length {}",
                shredded.len(),
                len
            );
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("VariantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let shredded_dtype = array.slots()[SHREDDED_SLOT]
            .as_ref()
            .map(|shredded| shredded.dtype().try_into())
            .transpose()?;
        Ok(Some(
            VariantMetadataProto { shredded_dtype }.encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],

        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<crate::array::ArrayParts<Self>> {
        vortex_ensure!(
            buffers.is_empty(),
            "VariantArray expects 0 buffers, got {}",
            buffers.len()
        );
        let proto = VariantMetadataProto::decode(metadata)?;
        let shredded_dtype = proto
            .shredded_dtype
            .as_ref()
            .map(|dtype| DType::from_proto(dtype, session))
            .transpose()?;
        vortex_ensure!(matches!(dtype, DType::Variant(_)), "Expected Variant DType");
        let expected_children = 1 + usize::from(shredded_dtype.is_some());
        vortex_ensure!(
            children.len() == expected_children,
            "Expected {} children, got {}",
            expected_children,
            children.len(),
        );
        let core_storage = children.get(0, dtype, len)?;
        let shredded = shredded_dtype
            .map(|dtype| children.get(1, &dtype, len))
            .transpose()?;
        Ok(
            crate::array::ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData)
                .with_slots(vec![Some(core_storage), shredded]),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match SLOT_NAMES.get(idx) {
            Some(name) => (*name).to_string(),
            None => vortex_panic!("VariantArray slot_name index {idx} out of bounds"),
        }
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

#[cfg(test)]
mod tests {}
