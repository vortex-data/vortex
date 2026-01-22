// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod canonical;
mod operations;
mod validity;

use std::ops::Range;

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
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::NotSupported;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;
use crate::vtable::VisitorVTable;

vtable!(Masked);

#[derive(Debug)]
pub struct MaskedVTable;

impl MaskedVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.masked");
}

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
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let child = array.child.slice(range.clone())?;
        let validity = array.validity.slice(range)?;

        Ok(Some(
            MaskedArray {
                child,
                validity,
                dtype: array.dtype.clone(),
                stats: ArrayStats::default(),
            }
            .into_array(),
        ))
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
        let validity_mask = array.validity_mask();

        // Fast path: all masked means result is all nulls.
        if validity_mask.all_false() {
            return ConstantArray::new(Scalar::null(array.dtype().as_nullable()), array.len())
                .into_array()
                .execute::<Canonical>(ctx);
        }

        // NB: We intentionally do NOT have a fast path for `validity_mask.all_true()`.
        // `MaskedArray`'s dtype is always `Nullable`, but the child has `NonNullable` `DType` (by
        // invariant). Simply returning the child's canonical would cause a dtype mismatch.
        // While we could manually convert the dtype, `mask_validity_canonical` is already O(1) for
        // `AllTrue` masks (no data copying), so there's no benefit.

        let child = array.child().clone().execute::<Canonical>(ctx)?;
        let canonical = mask_validity_canonical(child, &validity_mask, ctx)?;

        vortex_ensure!(
            canonical.as_ref().dtype() == array.dtype(),
            "Mask result dtype mismatch: expected {:?}, got {:?}",
            array.dtype(),
            canonical.as_ref().dtype()
        );
        vortex_ensure!(
            canonical.len() == array.len(),
            "Mask result length mismatch: expected {}, got {}",
            array.len(),
            canonical.len()
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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;
    use vortex_dtype::Nullability;
    use vortex_error::VortexError;

    use crate::ArrayContext;
    use crate::ArraySession;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::MaskedArray;
    use crate::arrays::MaskedVTable;
    use crate::arrays::PrimitiveArray;
    use crate::serde::ArrayParts;
    use crate::serde::SerializeOptions;
    use crate::validity::Validity;

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

        let ctx = ArrayContext::empty();
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

        let session = ArraySession::default();

        let parts = ArrayParts::try_from(concat).unwrap();
        let decoded = parts.decode(&dtype, len, &ctx, session.registry()).unwrap();

        assert!(decoded.is::<MaskedVTable>());
        assert_eq!(
            array.as_ref().display_values().to_string(),
            decoded.display_values().to_string()
        );
    }

    /// Regression test for issue #5989: execute_fast_path returns child with wrong dtype.
    ///
    /// When MaskedArray's validity mask is all true, returning the child's canonical form
    /// directly would cause a dtype mismatch because the child has NonNullable dtype while
    /// MaskedArray always has Nullable dtype.
    #[test]
    fn test_execute_with_all_valid_preserves_nullable_dtype() -> Result<(), VortexError> {
        // Create a MaskedArray with AllValid validity.

        // Child has NonNullable dtype, but MaskedArray's dtype is Nullable.
        let child = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        assert_eq!(child.dtype().nullability(), Nullability::NonNullable);

        let array = MaskedArray::try_new(child, Validity::AllValid)?;
        assert_eq!(array.dtype().nullability(), Nullability::Nullable);

        // Execute the array. This should produce a Canonical with Nullable dtype.
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result: Canonical = array.into_array().execute(&mut ctx)?;

        assert_eq!(
            result.as_ref().dtype().nullability(),
            Nullability::Nullable,
            "MaskedArray execute should produce Nullable dtype"
        );

        Ok(())
    }
}
