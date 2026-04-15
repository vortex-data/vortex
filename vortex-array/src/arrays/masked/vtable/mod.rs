// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
mod canonical;
mod operations;
mod validity;

use std::hash::Hasher;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyCanonical;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::Precision;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::validity_to_child;
use crate::arrays::ConstantArray;
use crate::arrays::masked::MaskedArrayExt;
use crate::arrays::masked::MaskedData;
use crate::arrays::masked::array::CHILD_SLOT;
use crate::arrays::masked::array::SLOT_NAMES;
use crate::arrays::masked::array::VALIDITY_SLOT;
use crate::arrays::masked::compute::rules::PARENT_RULES;
use crate::arrays::masked::mask_validity_canonical;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::require_child;
use crate::require_validity;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
/// A [`Masked`]-encoded Vortex array.
pub type MaskedArray = Array<Masked>;

#[derive(Clone, Debug)]
pub struct Masked;

impl ArrayHash for MaskedData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _precision: Precision) {}
}

impl ArrayEq for MaskedData {
    fn array_eq(&self, _other: &Self, _precision: Precision) -> bool {
        true
    }
}

impl VTable for Masked {
    type ArrayData = MaskedData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.masked");
        *ID
    }

    fn validate(
        &self,
        _data: &MaskedData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots[CHILD_SLOT].is_some(),
            "MaskedArray child slot must be present"
        );
        let child = slots[CHILD_SLOT]
            .as_ref()
            .vortex_expect("validated child slot");
        vortex_ensure!(child.len() == len, "MaskedArray child length mismatch");
        vortex_ensure!(
            child.dtype().as_nullable() == *dtype,
            "MaskedArray dtype does not match child and validity"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("MaskedArray has no buffers")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
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
        if !metadata.is_empty() {
            vortex_bail!(
                "MaskedArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        if !buffers.is_empty() {
            vortex_bail!("Expected 0 buffer, got {}", buffers.len());
        }

        vortex_ensure!(
            children.len() == 1 || children.len() == 2,
            "`MaskedArray::build` expects 1 or 2 children, got {}",
            children.len()
        );

        let child = children.get(0, &dtype.as_nonnullable(), len)?;

        let validity = if children.len() == 2 {
            let validity = children.get(1, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            Validity::from(dtype.nullability())
        };

        let validity_slot = validity_to_child(&validity, len);
        let data = MaskedData::try_new(
            len,
            child.all_valid(&mut LEGACY_SESSION.create_execution_ctx())?,
            validity,
        )?;
        Ok(
            crate::array::ArrayParts::new(self.clone(), dtype.clone(), len, data)
                .with_slots(vec![Some(child), validity_slot]),
        )
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let validity_mask = array.masked_validity().to_mask(array.len(), ctx)?;

        // Fast path: all masked means result is all nulls.
        if validity_mask.all_false() {
            return Ok(ExecutionResult::done(
                ConstantArray::new(Scalar::null(array.dtype().as_nullable()), array.len())
                    .into_array(),
            ));
        }

        // NB: We intentionally do NOT have a fast path for `validity_mask.all_true()`.
        // `MaskedArray`'s dtype is always `Nullable`, but the child has `NonNullable` `DType` (by
        // invariant). Simply returning the child's canonical would cause a dtype mismatch.
        // While we could manually convert the dtype, `mask_validity_canonical` is already O(1) for
        // `AllTrue` masks (no data copying), so there's no benefit.

        let array = require_child!(array, array.child(), CHILD_SLOT => AnyCanonical);
        require_validity!(array, VALIDITY_SLOT);

        let child = Canonical::from(array.child().as_::<AnyCanonical>());
        Ok(ExecutionResult::done(
            mask_validity_canonical(child, &validity_mask, ctx)?.into_array(),
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexError;
    use vortex_session::registry::ReadContext;

    use crate::ArrayContext;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::Masked;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::Nullability;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
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
            .clone()
            .into_array()
            .serialize(&ctx, &LEGACY_SESSION, &SerializeOptions::default())
            .unwrap();

        // Concat into a single buffer.
        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = SerializedArray::try_from(concat).unwrap();
        let decoded = parts
            .decode(
                &dtype,
                len,
                &ReadContext::new(ctx.to_ids()),
                &LEGACY_SESSION,
            )
            .unwrap();

        assert!(decoded.is::<Masked>());
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
            result.dtype().nullability(),
            Nullability::Nullable,
            "MaskedArray execute should produce Nullable dtype"
        );

        Ok(())
    }
}
