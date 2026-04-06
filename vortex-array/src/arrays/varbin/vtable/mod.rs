// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::varbin::VarBinData;
use crate::arrays::varbin::array::NUM_SLOTS;
use crate::arrays::varbin::array::SLOT_NAMES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
mod canonical;
mod kernel;
mod operations;
mod validity;

use canonical::varbin_to_canonical;
use kernel::PARENT_KERNELS;
use vortex_session::VortexSession;

use crate::Precision;
use crate::arrays::varbin::compute::rules::PARENT_RULES;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::ArrayStats;

vtable!(VarBin, VarBin, VarBinData);

#[derive(Clone, prost::Message)]
pub struct VarBinMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub(crate) offsets_ptype: i32,
}

impl VTable for VarBin {
    type ArrayData = VarBinData;

    type Metadata = ProstMetadata<VarBinMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn vtable(_array: &VarBinData) -> &Self {
        &VarBin
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &VarBinData) -> usize {
        array.offsets().len().saturating_sub(1)
    }

    fn dtype(array: &VarBinData) -> &DType {
        &array.dtype
    }

    fn stats(array: &VarBinData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(array: &VarBinData, state: &mut H, precision: Precision) {
        array.bytes().array_hash(state, precision);
        array.offsets().array_hash(state, precision);
        array.validity().array_hash(state, precision);
    }

    fn array_eq(array: &VarBinData, other: &VarBinData, precision: Precision) -> bool {
        array.bytes().array_eq(other.bytes(), precision)
            && array.offsets().array_eq(other.offsets(), precision)
            && array.validity().array_eq(&other.validity(), precision)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.bytes_handle().clone(),
            _ => vortex_panic!("VarBinArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("bytes".to_string()),
            _ => vortex_panic!("VarBinArray buffer_name index {idx} out of bounds"),
        }
    }

    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<VarBinData> {
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

        VarBinData::try_new(offsets, bytes, dtype.clone(), validity)
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
            "VarBinArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
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

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            varbin_to_canonical(array.as_view(), ctx)?.into_array(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct VarBin;

impl VarBin {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.varbin");
}
