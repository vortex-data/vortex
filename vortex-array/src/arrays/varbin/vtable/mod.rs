// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::DeserializeMetadata;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::arrays::varbin::VarBinArray;
use crate::buffer::BufferHandle;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;

mod array;
mod canonical;
mod kernel;
mod operations;
mod validity;
mod visitor;

use canonical::varbin_to_canonical;
use vortex_session::VortexSession;

use crate::arrays::varbin::compute::rules::PARENT_RULES;

vtable!(VarBin);

#[derive(Clone, prost::Message)]
pub struct VarBinMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub(crate) offsets_ptype: i32,
}

impl VTable for VarBinVTable {
    type Array = VarBinArray;

    type Metadata = ProstMetadata<VarBinMetadata>;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
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
        let bytes = buffers[0].clone();

        VarBinArray::try_new_from_handle(offsets, bytes, dtype.clone(), validity)
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
        kernel::PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(varbin_to_canonical(array, ctx)?.into_array())
    }
}

#[derive(Debug)]
pub struct VarBinVTable;

impl VarBinVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.varbin");
}
