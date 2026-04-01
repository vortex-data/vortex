// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use kernel::PARENT_KERNELS;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::DeserializeMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::arrays::BoolArray;
use crate::arrays::bool::array::NUM_SLOTS;
use crate::arrays::bool::array::SLOT_NAMES;
use crate::arrays::bool::array::VALIDITY_SLOT;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;
mod canonical;
mod kernel;
mod operations;
mod validity;

use std::hash::Hash;

use crate::Precision;
use crate::arrays::bool::compute::rules::RULES;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::StatsSetRef;
use crate::vtable::ArrayId;

vtable!(Bool);

#[derive(prost::Message)]
pub struct BoolMetadata {
    // The offset in bits must be <8
    #[prost(uint32, tag = "1")]
    pub offset: u32,
}

impl VTable for Bool {
    type Array = BoolArray;

    type Metadata = ProstMetadata<BoolMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;

    fn vtable(_array: &Self::Array) -> &Self {
        &Bool
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &BoolArray) -> usize {
        array.len
    }

    fn dtype(array: &BoolArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &BoolArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &BoolArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.to_bit_buffer().array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &BoolArray, other: &BoolArray, precision: Precision) -> bool {
        if array.dtype != other.dtype {
            return false;
        }
        array
            .to_bit_buffer()
            .array_eq(&other.to_bit_buffer(), precision)
            && array.validity.array_eq(&other.validity, precision)
    }

    fn nbuffers(_array: &BoolArray) -> usize {
        1
    }

    fn buffer(array: &BoolArray, idx: usize) -> BufferHandle {
        match idx {
            0 => array.bits.clone(),
            _ => vortex_panic!("BoolArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: &BoolArray, idx: usize) -> Option<String> {
        match idx {
            0 => Some("bits".to_string()),
            _ => None,
        }
    }

    fn metadata(array: &BoolArray) -> VortexResult<Self::Metadata> {
        assert!(array.offset < 8, "Offset must be <8, got {}", array.offset);
        Ok(ProstMetadata(BoolMetadata {
            offset: u32::try_from(array.offset).vortex_expect("checked"),
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let metadata = <Self::Metadata as DeserializeMetadata>::deserialize(bytes)?;
        Ok(ProstMetadata(metadata))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
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

        let buffer = buffers[0].clone();

        BoolArray::try_new_from_handle(buffer, metadata.offset as usize, len, validity)
    }

    fn slots(array: &BoolArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &BoolArray, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut BoolArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "BoolArray expects {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.validity = match &slots[VALIDITY_SLOT] {
            Some(arr) => Validity::Array(arr.clone()),
            None => Validity::from(array.dtype().nullability()),
        };
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

#[derive(Clone, Debug)]
pub struct Bool;

impl Bool {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.bool");
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBufferMut;
    use vortex_session::registry::ReadContext;

    use crate::ArrayContext;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::arrays::BoolArray;
    use crate::assert_arrays_eq;
    use crate::serde::ArrayParts;
    use crate::serde::SerializeOptions;

    #[test]
    fn test_nullable_bool_serde_roundtrip() {
        let array = BoolArray::from_iter([Some(true), None, Some(false), None]);
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
