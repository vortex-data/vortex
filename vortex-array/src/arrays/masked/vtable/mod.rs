// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
mod canonical;
mod operations;
mod validity;

use std::hash::Hash;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyMetadata;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::ConstantArray;
use crate::arrays::MaskedArray;
use crate::arrays::masked::array::NUM_SLOTS;
use crate::arrays::masked::array::SLOT_NAMES;
use crate::arrays::masked::array::VALIDITY_SLOT;
use crate::arrays::masked::compute::rules::PARENT_RULES;
use crate::arrays::masked::mask_validity_canonical;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;
vtable!(Masked);

#[derive(Clone, Debug)]
pub struct Masked;

impl Masked {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.masked");
}

impl VTable for Masked {
    type Array = MaskedArray;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;

    fn vtable(_array: &Self::Array) -> &Self {
        &Masked
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &MaskedArray) -> usize {
        array.child().len()
    }

    fn dtype(array: &MaskedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &MaskedArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &MaskedArray, state: &mut H, precision: Precision) {
        array.child().array_hash(state, precision);
        array.validity.array_hash(state, precision);
        array.dtype.hash(state);
    }

    fn array_eq(array: &MaskedArray, other: &MaskedArray, precision: Precision) -> bool {
        array.child().array_eq(other.child(), precision)
            && array.validity.array_eq(&other.validity, precision)
            && array.dtype == other.dtype
    }

    fn nbuffers(_array: &Self::Array) -> usize {
        0
    }

    fn buffer(_array: &Self::Array, _idx: usize) -> BufferHandle {
        vortex_panic!("MaskedArray has no buffers")
    }

    fn buffer_name(_array: &Self::Array, _idx: usize) -> Option<String> {
        None
    }

    fn metadata(_array: &MaskedArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
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

        MaskedArray::try_new(child, validity)
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let validity_mask = array.validity_mask()?;

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

        let child = array.child().clone().execute::<Canonical>(ctx)?;
        Ok(ExecutionResult::done(
            mask_validity_canonical(child, &validity_mask, ctx)?.into_array(),
        ))
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn slots(array: &MaskedArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &MaskedArray, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut MaskedArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "MaskedArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.validity = match &slots[VALIDITY_SLOT] {
            Some(arr) => Validity::Array(arr.clone()),
            None => Validity::from(array.dtype.nullability()),
        };
        array.slots = slots;
        Ok(())
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
            .clone()
            .into_array()
            .serialize(&ctx, &SerializeOptions::default())
            .unwrap();

        // Concat into a single buffer.
        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = ArrayParts::try_from(concat).unwrap();
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
            result.as_ref().dtype().nullability(),
            Nullability::Nullable,
            "MaskedArray execute should produce Nullable dtype"
        );

        Ok(())
    }
}
