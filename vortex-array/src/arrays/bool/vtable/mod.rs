// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_vector::bool::BoolVector;

use crate::ArrayRef;
use crate::DeserializeMetadata;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::arrays::BoolArray;
use crate::kernel::BindCtx;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;

mod array;
mod canonical;
mod operations;
pub mod operator;
mod validity;
mod visitor;

pub use operator::BoolMaskedValidityRule;

use crate::kernel::KernelRef;
use crate::kernel::ready;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;

vtable!(Bool);

#[derive(prost::Message)]
pub struct BoolMetadata {
    // The offset in bits must be <8
    #[prost(uint32, tag = "1")]
    pub offset: u32,
}

impl VTable for BoolVTable {
    type Array = BoolArray;

    type Metadata = ProstMetadata<BoolMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.bool")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        BoolVTable.as_vtable()
    }

    fn metadata(array: &BoolArray) -> VortexResult<Self::Metadata> {
        let bit_offset = array.bit_buffer().offset();
        assert!(bit_offset < 8, "Offset must be <8, got {bit_offset}");
        Ok(ProstMetadata(BoolMetadata {
            offset: u32::try_from(bit_offset).vortex_expect("checked"),
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(bytes: &[u8]) -> VortexResult<Self::Metadata> {
        let metadata = <Self::Metadata as DeserializeMetadata>::deserialize(bytes)?;
        Ok(ProstMetadata(metadata))
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<BoolArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }

        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        let buffer = buffers[0].clone().try_to_bytes()?;
        let bits = BitBuffer::new_with_offset(buffer, len, metadata.offset as usize);

        BoolArray::try_new(bits, validity)
    }

    fn bind_kernel(array: &Self::Array, _ctx: &mut BindCtx) -> VortexResult<KernelRef> {
        Ok(ready(
            BoolVector::new(array.bit_buffer().clone(), array.validity_mask()).into(),
        ))
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() <= 1,
            "BoolArray can have at most 1 child (validity), got {}",
            children.len()
        );

        array.validity = if children.is_empty() {
            Validity::from(array.dtype().nullability())
        } else {
            Validity::Array(children.into_iter().next().vortex_expect("checked"))
        };

        Ok(())
    }
}

#[derive(Debug)]
pub struct BoolVTable;
