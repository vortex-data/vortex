// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::DeserializeMetadata;
use crate::ExecutionCtx;
use crate::ExecutionStep;
use crate::IntoArray;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::arrays::VarBinArray;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;
use crate::vtable::validity_nchildren;
use crate::vtable::validity_to_child;
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

    fn nchildren(array: &VarBinArray) -> usize {
        1 + validity_nchildren(&array.validity)
    }

    fn child(array: &VarBinArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.offsets().clone(),
            1 => validity_to_child(&array.validity, array.len())
                .vortex_expect("VarBinArray validity child out of bounds"),
            _ => vortex_panic!("VarBinArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &VarBinArray, idx: usize) -> String {
        match idx {
            0 => "offsets".to_string(),
            1 => "validity".to_string(),
            _ => vortex_panic!("VarBinArray child_name index {idx} out of bounds"),
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

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        match children.len() {
            1 => {
                let [offsets]: [ArrayRef; 1] = children
                    .try_into()
                    .map_err(|_| vortex_err!("Failed to convert children to array"))?;
                array.offsets = offsets;
            }
            2 => {
                let [offsets, validity]: [ArrayRef; 2] = children
                    .try_into()
                    .map_err(|_| vortex_err!("Failed to convert children to array"))?;
                array.offsets = offsets;
                array.validity = Validity::Array(validity);
            }
            _ => vortex_bail!(
                "VarBinArray expects 1 or 2 children (offsets, validity?), got {}",
                children.len()
            ),
        }
        Ok(())
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        Ok(ExecutionStep::Done(
            varbin_to_canonical(array, ctx)?.into_array(),
        ))
    }
}

#[derive(Debug)]
pub struct VarBin;

impl VarBin {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.varbin");
}
