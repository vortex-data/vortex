// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod canonical;
mod operations;
mod operator;
mod validity;

use vortex_buffer::BufferHandle;
use vortex_compute::mask::MaskValidity;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Vector;

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayOperator;
use crate::EmptyMetadata;
use crate::arrays::masked::MaskedArray;
use crate::execution::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;
use crate::vtable::VisitorVTable;

vtable!(Masked);

#[derive(Debug)]
pub struct MaskedVTable;

impl VisitorVTable<MaskedVTable> for MaskedVTable {
    fn visit_buffers(_array: &MaskedArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &MaskedArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", array.child.as_ref());
        visitor.visit_validity(&array.validity, array.child.len());
    }
}

impl VTable for MaskedVTable {
    type Array = MaskedArray;

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
        ArrayId::new_ref("vortex.masked")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        MaskedVTable.as_vtable()
    }

    fn metadata(_array: &MaskedArray) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<MaskedArray> {
        if !buffers.is_empty() {
            vortex_bail!("Expected 0 buffer, got {}", buffers.len());
        }

        let child = children.get(0, &dtype.as_nonnullable(), len)?;

        let validity = if children.len() == 1 {
            Validity::from(dtype.nullability())
        } else if children.len() == 2 {
            let validity = children.get(1, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!(
                "`MaskedArray::build` expects 1 or 2 children, got {}",
                children.len()
            );
        };

        MaskedArray::try_new(child, validity)
    }

    fn execute(array: &Self::Array, ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        let vector = array.child().execute_batch(ctx)?;
        Ok(MaskValidity::mask_validity(vector, &array.validity_mask()))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;

    use crate::ArrayContext;
    use crate::IntoArray;
    use crate::MaskedVTable;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::serde::ArrayParts;
    use crate::serde::SerializeOptions;
    use crate::validity::Validity;
    use crate::vtable::ArrayVTableExt;

    #[rstest]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            Validity::AllValid
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            Validity::from_iter([true, true, false, true, false])
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter(0..100).into_array(),
            Validity::from_iter((0..100).map(|i| i % 3 != 0))
        ).unwrap()
    )]
    fn test_serde_roundtrip(#[case] array: MaskedArray) {
        let dtype = array.dtype().clone();
        let len = array.len();
        let ctx = ArrayContext::empty().with(MaskedVTable.as_vtable());

        let serialized = array
            .to_array()
            .serialize(&ctx, &SerializeOptions::default())
            .unwrap();

        // Concat into a single buffer.
        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = ArrayParts::try_from(concat).unwrap();
        let decoded = parts.decode(&ctx, &dtype, len).unwrap();

        assert!(decoded.is::<MaskedVTable>());
        assert_eq!(
            array.as_ref().display_values().to_string(),
            decoded.display_values().to_string()
        );
    }
}
