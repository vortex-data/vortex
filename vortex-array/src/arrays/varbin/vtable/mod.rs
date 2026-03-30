// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::DeserializeMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::arrays::VarBinArray;
use crate::arrays::varbin::array::NUM_SLOTS;
use crate::arrays::varbin::array::SLOT_NAMES;
use crate::arrays::varbin::array::VALIDITY_SLOT;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;
mod canonical;
mod kernel;
mod operations;
mod validity;
use std::hash::Hash;

use canonical::varbin_to_canonical;
use kernel::PARENT_KERNELS;
use vortex_session::VortexSession;

use crate::Precision;
use crate::arrays::varbin::compute::rules::PARENT_RULES;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::StatsSetRef;

vtable!(VarBin);

#[derive(Clone, prost::Message)]
pub struct VarBinMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub(crate) offsets_ptype: i32,
}

impl VTable for VarBin {
    type Array = VarBinArray;

    type Metadata = ProstMetadata<VarBinMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    fn vtable(_array: &Self::Array) -> &Self {
        &VarBin
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &VarBinArray) -> usize {
        array.offsets().len().saturating_sub(1)
    }

    fn dtype(array: &VarBinArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &VarBinArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &VarBinArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.bytes().array_hash(state, precision);
        array.offsets().array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &VarBinArray, other: &VarBinArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.bytes().array_eq(other.bytes(), precision)
            && array.offsets().array_eq(other.offsets(), precision)
            && array.validity.array_eq(&other.validity, precision)
    }

    fn nbuffers(_array: &VarBinArray) -> usize {
        1
    }

    fn buffer(array: &VarBinArray, idx: usize) -> BufferHandle {
        match idx {
            0 => array.bytes_handle().clone(),
            _ => vortex_panic!("VarBinArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: &VarBinArray, idx: usize) -> Option<String> {
        match idx {
            0 => Some("bytes".to_string()),
            _ => vortex_panic!("VarBinArray buffer_name index {idx} out of bounds"),
        }
    }

    fn metadata(array: &VarBinArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(VarBinMetadata {
            offsets_ptype: PType::try_from(array.offsets().dtype())
                .vortex_expect("Must be a valid PType") as i32,
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
        Ok(ProstMetadata(ProstMetadata::<VarBinMetadata>::deserialize(
            bytes,
        )?))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<VarBinArray> {
        let validity = if children.len() == 1 {
            Validity::from(dtype.nullability())
        } else if children.len() == 2 {
            let validity = children.get(1, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 1 or 2 children, got {}", children.len());
        };

        let offsets = children.get(
            0,
            &DType::Primitive(metadata.offsets_ptype(), Nullability::NonNullable),
            len + 1,
        )?;

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let bytes = buffers[0].clone().try_to_host_sync()?;

        VarBinArray::try_new(offsets, bytes, dtype.clone(), validity)
    }

    fn slots(array: &VarBinArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &VarBinArray, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut VarBinArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "VarBinArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.validity = match &slots[VALIDITY_SLOT] {
            Some(arr) => Validity::Array(arr.clone()),
            None => Validity::from(array.dtype.nullability()),
        };
        array.slots = slots;
        Ok(())
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            varbin_to_canonical(&array, ctx)?.into_array(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct VarBin;

impl VarBin {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.varbin");
}
