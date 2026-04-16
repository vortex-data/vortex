// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use kernel::PARENT_KERNELS;
use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::bool::BoolData;
use crate::arrays::bool::array::SLOT_NAMES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
mod canonical;
mod kernel;
mod operations;
mod validity;

use vortex_session::registry::CachedId;

use crate::Precision;
use crate::array::ArrayId;
use crate::arrays::bool::compute::rules::RULES;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;

/// A [`Bool`]-encoded Vortex array.
pub type BoolArray = Array<Bool>;

#[derive(prost::Message)]
pub struct BoolMetadata {
    // The offset in bits must be <8
    #[prost(uint32, tag = "1")]
    pub offset: u32,
}

impl ArrayHash for BoolData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.bits.array_hash(state, precision);
        self.offset.hash(state);
    }
}

impl ArrayEq for BoolData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.offset == other.offset && self.bits.array_eq(&other.bits, precision)
    }
}

impl VTable for Bool {
    type ArrayData = BoolData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.bool");
        *ID
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.bits.clone(),
            _ => vortex_panic!("BoolArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("bits".to_string()),
            _ => None,
        }
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        assert!(array.offset < 8, "Offset must be <8, got {}", array.offset);
        Ok(Some(
            BoolMetadata {
                offset: u32::try_from(array.offset).vortex_expect("checked"),
            }
            .encode_to_vec(),
        ))
    }

    fn validate(
        &self,
        data: &BoolData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Bool(nullability) = dtype else {
            vortex_bail!("Expected bool dtype, got {dtype:?}");
        };
        vortex_ensure!(
            data.bits.len() * 8 >= data.offset + len,
            "BoolArray buffer with offset {} cannot back outer length {} (buffer bits = {})",
            data.offset,
            len,
            data.bits.len() * 8
        );

        let validity = crate::array::child_to_validity(&slots[0], *nullability);
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == len,
                "BoolArray validity len {} does not match outer length {}",
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
        let metadata = BoolMetadata::decode(metadata)?;
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

        let buffer = buffers[0].clone();
        let slots = BoolData::make_slots(&validity, len);
        let data = BoolData::try_new_from_handle(buffer, metadata.offset as usize, len, validity)?;
        Ok(crate::array::ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

#[derive(Clone, Debug)]
pub struct Bool;

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBufferMut;
    use vortex_session::registry::ReadContext;

    use crate::ArrayContext;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::arrays::BoolArray;
    use crate::assert_arrays_eq;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;

    #[test]
    fn test_nullable_bool_serde_roundtrip() {
        let array = BoolArray::from_iter([Some(true), None, Some(false), None]);
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
