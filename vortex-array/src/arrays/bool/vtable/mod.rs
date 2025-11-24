// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_vector::Vector;
use vortex_vector::bool::BoolVector;

use crate::arrays::BoolArray;
use crate::execution::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{ArrayVTableExt, NotSupported, VTable, ValidityVTableFromValidityHelper};
use crate::{DeserializeMetadata, ProstMetadata, SerializeMetadata, vtable};

mod array;
mod canonical;
mod operations;
pub mod operator;
mod validity;
mod visitor;

pub use operator::BoolMaskedValidityRule;

use crate::vtable::{ArrayId, ArrayVTable};

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
    type OperatorVTable = Self;

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
        buffers: &[ByteBuffer],
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

        BoolArray::try_new(buffers[0].clone(), metadata.offset as usize, len, validity)
    }

    fn execute(array: &Self::Array, _ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        Ok(BoolVector::new(array.bit_buffer().clone(), array.validity_mask()).into())
    }
}

#[derive(Clone, Debug)]
pub struct BoolVTable;
