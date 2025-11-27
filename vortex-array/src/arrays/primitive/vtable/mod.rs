// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_dtype::PType;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Vector;
use vortex_vector::primitive::PVector;

use crate::EmptyMetadata;
use crate::arrays::PrimitiveArray;
use crate::execution::ExecutionCtx;
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

pub use operator::PrimitiveMaskedValidityRule;

use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;

vtable!(Primitive);

impl VTable for PrimitiveVTable {
    type Array = PrimitiveArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type OperatorVTable = Self;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.primitive")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        PrimitiveVTable.as_vtable()
    }

    fn metadata(_array: &PrimitiveArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(_buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<PrimitiveArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let buffer = buffers[0].clone().try_to_bytes()?;

        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        let ptype = PType::try_from(dtype)?;

        if !buffer.is_aligned(Alignment::new(ptype.byte_width())) {
            vortex_bail!(
                "Buffer is not aligned to {}-byte boundary",
                ptype.byte_width()
            );
        }
        if buffer.len() != ptype.byte_width() * len {
            vortex_bail!(
                "Buffer length {} does not match expected length {} for {}, {}",
                buffer.len(),
                ptype.byte_width() * len,
                ptype.byte_width(),
                len,
            );
        }

        match_each_native_ptype!(ptype, |P| {
            let buffer = Buffer::<P>::from_byte_buffer(buffer);
            Ok(PrimitiveArray::new(buffer, validity))
        })
    }

    fn execute(array: &Self::Array, _ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        Ok(match_each_native_ptype!(array.ptype(), |T| {
            PVector::new(array.buffer::<T>(), array.validity_mask()).into()
        }))
    }
}

#[derive(Debug)]
pub struct PrimitiveVTable;
