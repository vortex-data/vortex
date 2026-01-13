// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod canonical;
mod operations;
mod validity;

use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_scalar::Scalar;

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyMetadata;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::masked::MaskedArray;
use crate::arrays::masked::mask_validity_canonical;
use crate::buffer::BufferHandle;
use crate::executor::ExecutionCtx;
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
        visitor.visit_child("child", &array.child);
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

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        if let Some(canonical) = execute_fast_path(array, ctx)? {
            return Ok(canonical);
        }

        let child = array.child().clone().execute::<Canonical>(ctx)?;
        let canonical = mask_validity_canonical(child, &array.validity_mask());

        vortex_ensure!(
            canonical.as_ref().dtype() == array.dtype(),
            "Mask result dtype mismatch: expected {:?}, got {:?}",
            array.dtype(),
            canonical.as_ref().dtype()
        );
        vortex_ensure!(
            canonical.as_ref().len() == array.len(),
            "Mask result length mismatch: expected {}, got {}",
            array.len(),
            canonical.as_ref().len()
        );

        Ok(canonical)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1 || children.len() == 2,
            "MaskedArray expects 1 or 2 children, got {}",
            children.len()
        );

        let mut iter = children.into_iter();
        let child = iter
            .next()
            .vortex_expect("children length already validated");
        let validity = if let Some(validity_array) = iter.next() {
            Validity::Array(validity_array)
        } else {
            Validity::from(array.dtype.nullability())
        };

        let new_array = MaskedArray::try_new(child, validity)?;
        *array = new_array;
        Ok(())
    }
}

/// Check for fast-path execution conditions.
pub(super) fn execute_fast_path(
    array: &MaskedArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Canonical>> {
    let validity_mask = array.validity_mask();

    // All valid - no masking needed
    if validity_mask.all_true() {
        return Ok(Some(array.child.clone().execute(ctx)?));
    }

    // All masked - result is all nulls
    if validity_mask.all_false() {
        return Ok(Some(
            ConstantArray::new(Scalar::null(array.dtype().as_nullable()), array.len())
                .into_array()
                .execute::<Canonical>(ctx)?,
        ));
    }

    // Child is already all nulls - masking has no effect
    if array.child.all_invalid() {
        return Ok(Some(array.child.clone().execute(ctx)?));
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;

    use crate::ArrayContext;
    use crate::IntoArray;
    use crate::arrays::MaskedArray;
    use crate::arrays::MaskedVTable;
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
