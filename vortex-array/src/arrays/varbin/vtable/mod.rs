// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrays::varbin::VarBinData;
use crate::arrays::varbin::array::NUM_SLOTS;
use crate::arrays::varbin::array::OFFSETS_SLOT;
use crate::arrays::varbin::array::SLOT_NAMES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
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

/// A [`VarBin`]-encoded Vortex array.
pub type VarBinArray = Array<VarBin>;

#[derive(Clone, prost::Message)]
pub struct VarBinMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub(crate) offsets_ptype: i32,
}

impl ArrayHash for VarBinData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.bytes().array_hash(state, precision);
    }
}

impl ArrayEq for VarBinData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.bytes().array_eq(other.bytes(), precision)
    }
}

impl VTable for VarBin {
    type TypedArrayData = VarBinData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;
    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.varbin");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn validate(
        &self,
        _data: &VarBinData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "VarBinArray expected {NUM_SLOTS} slots, found {}",
            slots.len()
        );
        let offsets = slots[OFFSETS_SLOT]
            .as_ref()
            .vortex_expect("VarBinArray offsets slot");
        vortex_ensure!(
            offsets.len().saturating_sub(1) == len,
            "VarBinArray length {} does not match outer length {}",
            offsets.len().saturating_sub(1),
            len
        );
        vortex_ensure!(
            matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
            "VarBinArray dtype must be binary or utf8, got {dtype}"
        );
        Ok(())
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

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            VarBinMetadata {
                offsets_ptype: PType::try_from(array.offsets().dtype())
                    .vortex_expect("Must be a valid PType") as i32,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],

        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<crate::array::ArrayParts<Self>> {
        let metadata = VarBinMetadata::decode(metadata)?;
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

        let data = VarBinData::try_build(offsets.clone(), bytes, dtype.clone(), validity.clone())?;
        let slots = VarBinData::make_slots(offsets, &validity, len);
        Ok(crate::array::ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
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
