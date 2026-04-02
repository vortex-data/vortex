// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use kernel::PARENT_KERNELS;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::primitive::PrimitiveData;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
mod kernel;
mod operations;
mod validity;

use std::hash::Hasher;

use vortex_buffer::Alignment;
use vortex_session::VortexSession;

use crate::Precision;
use crate::array::ArrayId;
use crate::arrays::primitive::array::NUM_SLOTS;
use crate::arrays::primitive::array::SLOT_NAMES;
use crate::arrays::primitive::compute::rules::RULES;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::ArrayStats;

vtable!(Primitive, Primitive, PrimitiveData);

impl VTable for Primitive {
    type ArrayData = PrimitiveData;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(_array: &Self::ArrayData) -> &Self {
        &Primitive
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &PrimitiveData) -> usize {
        array.buffer_handle().len() / array.ptype().byte_width()
    }

    fn dtype(array: &PrimitiveData) -> &DType {
        &array.dtype
    }

    fn stats(array: &PrimitiveData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: Hasher>(array: &PrimitiveData, state: &mut H, precision: Precision) {
        array.buffer.array_hash(state, precision);
        array.validity().array_hash(state, precision);
    }

    fn array_eq(array: &PrimitiveData, other: &PrimitiveData, precision: Precision) -> bool {
        array.buffer.array_eq(&other.buffer, precision)
            && array.validity().array_eq(&other.validity(), precision)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.buffer_handle().clone(),
            _ => vortex_panic!("PrimitiveArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("values".to_string()),
            _ => None,
        }
    }

    fn metadata(_array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<PrimitiveData> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let buffer = buffers[0].clone();

        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        let ptype = PType::try_from(dtype)?;

        vortex_ensure!(
            buffer.is_aligned_to(Alignment::new(ptype.byte_width())),
            "Misaligned buffer cannot be used to build PrimitiveArray of {ptype}"
        );

        if buffer.len() != ptype.byte_width() * len {
            vortex_bail!(
                "Buffer length {} does not match expected length {} for {}, {}",
                buffer.len(),
                ptype.byte_width() * len,
                ptype.byte_width(),
                len,
            );
        }

        vortex_ensure!(
            buffer.is_aligned_to(Alignment::new(ptype.byte_width())),
            "PrimitiveArray::build: Buffer (align={}) must be aligned to {}",
            buffer.alignment(),
            ptype.byte_width()
        );

        // SAFETY: checked ahead of time
        unsafe {
            Ok(PrimitiveData::new_unchecked_from_handle(
                buffer, ptype, validity,
            ))
        }
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "PrimitiveArray expects {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );

        array.slots = slots;
        Ok(())
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

#[derive(Clone, Debug)]
pub struct Primitive;

impl Primitive {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.primitive");
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_session::registry::ReadContext;

    use crate::ArrayContext;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::serde::ArrayParts;
    use crate::serde::SerializeOptions;
    use crate::validity::Validity;

    #[test]
    fn test_nullable_primitive_serde_roundtrip() {
        let array = PrimitiveArray::new(
            buffer![1i32, 2, 3, 4],
            Validity::from_iter([true, false, true, false]),
        );
        let dtype = array.dtype().clone();
        let len = array.len();

        let ctx = ArrayContext::empty();
        let serialized = array
            .clone()
            .into_array()
            .serialize(&ctx, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let parts = ArrayParts::try_from(concat.freeze()).unwrap();
        let decoded = parts
            .decode(
                &dtype,
                len,
                &ReadContext::new(ctx.to_ids()),
                &LEGACY_SESSION,
            )
            .unwrap();

        assert_arrays_eq!(decoded, array);
    }
}
