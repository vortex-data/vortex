// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use kernel::PARENT_KERNELS;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayRef;
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
mod kernel;
mod operations;
mod validity;

use std::hash::Hasher;

use vortex_buffer::Alignment;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::EqMode;
use crate::array::ArrayId;
use crate::arrays::primitive::array::SLOT_NAMES;
use crate::arrays::primitive::compute::rules::RULES;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;

/// A [`Primitive`]-encoded Vortex array.
pub type PrimitiveArray = Array<Primitive>;

impl ArrayHash for PrimitiveData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.buffer.array_hash(state, accuracy);
    }
}

impl ArrayEq for PrimitiveData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.buffer.array_eq(&other.buffer, accuracy)
    }
}

impl VTable for Primitive {
    type TypedArrayData = PrimitiveData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.primitive");
        *ID
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

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn validate(
        &self,
        data: &PrimitiveData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Primitive(_, nullability) = dtype else {
            vortex_bail!("Expected primitive dtype, got {dtype:?}");
        };
        vortex_ensure!(
            data.len() == len,
            "PrimitiveArray length {} does not match outer length {}",
            data.len(),
            len
        );
        let validity = crate::array::child_to_validity(slots[0].as_ref(), *nullability);
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == len,
                "PrimitiveArray validity len {} does not match outer length {}",
                validity_len,
                len
            );
        }

        Ok(())
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
                "PrimitiveArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
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
        let slots = PrimitiveData::make_slots(&validity, len);
        let data = unsafe { PrimitiveData::new_unchecked_from_handle(buffer, ptype, validity) };
        Ok(crate::array::ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
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
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
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
            .serialize(&ctx, &LEGACY_SESSION, &SerializeOptions::default())
            .unwrap();

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let parts = SerializedArray::try_from(concat.freeze()).unwrap();
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
