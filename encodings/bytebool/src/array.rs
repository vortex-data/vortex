// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::arrays::BoolArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityHelper;
use vortex_array::vtable::ValidityVTableFromValidityHelper;
use vortex_array::vtable::validity_nchildren;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::kernel::PARENT_KERNELS;

vtable!(ByteBool);

impl VTable for ByteBool {
    type Array = ByteBoolArray;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;

    fn vtable(_array: &Self::Array) -> &Self {
        &ByteBool
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ByteBoolArray) -> usize {
        array.buffer.len()
    }

    fn dtype(array: &ByteBoolArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ByteBoolArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &ByteBoolArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.buffer.array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &ByteBoolArray, other: &ByteBoolArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.buffer.array_eq(&other.buffer, precision)
            && array.validity.array_eq(&other.validity, precision)
    }

    fn nbuffers(_array: &ByteBoolArray) -> usize {
        1
    }

    fn buffer(array: &ByteBoolArray, idx: usize) -> BufferHandle {
        match idx {
            0 => array.buffer().clone(),
            _ => vortex_panic!("ByteBoolArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: &ByteBoolArray, idx: usize) -> Option<String> {
        match idx {
            0 => Some("values".to_string()),
            _ => vortex_panic!("ByteBoolArray buffer_name index {idx} out of bounds"),
        }
    }

    fn nchildren(array: &ByteBoolArray) -> usize {
        validity_nchildren(array.validity())
    }

    fn child(array: &ByteBoolArray, idx: usize) -> ArrayRef {
        match idx {
            0 => validity_to_child(array.validity(), array.len())
                .vortex_expect("ByteBoolArray validity child out of bounds"),
            _ => vortex_panic!("ByteBoolArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &ByteBoolArray, idx: usize) -> String {
        match idx {
            0 => "validity".to_string(),
            _ => vortex_panic!("ByteBoolArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(_array: &ByteBoolArray) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<ByteBoolArray> {
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let buffer = buffers[0].clone();

        Ok(ByteBoolArray::new(buffer, validity))
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() <= 1,
            "ByteBoolArray expects at most 1 child (validity), got {}",
            children.len()
        );

        array.validity = if children.is_empty() {
            Validity::from(array.dtype.nullability())
        } else {
            Validity::Array(children.into_iter().next().vortex_expect("checked"))
        };

        Ok(())
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::rules::RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Arc<Self::Array>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let boolean_buffer = BitBuffer::from(array.as_slice());
        let validity = array.validity().clone();
        Ok(ExecutionResult::done(
            BoolArray::new(boolean_buffer, validity).into_array(),
        ))
    }

    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

#[derive(Clone, Debug)]
pub struct ByteBoolArray {
    dtype: DType,
    buffer: BufferHandle,
    validity: Validity,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct ByteBool;

impl ByteBool {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.bytebool");
}

impl ByteBoolArray {
    pub fn new(buffer: BufferHandle, validity: Validity) -> Self {
        let length = buffer.len();
        if let Some(vlen) = validity.maybe_len()
            && length != vlen
        {
            vortex_panic!(
                "Buffer length ({}) does not match validity length ({})",
                length,
                vlen
            );
        }
        Self {
            dtype: DType::Bool(validity.nullability()),
            buffer,
            validity,
            stats_set: Default::default(),
        }
    }

    // TODO(ngates): deprecate construction from vec
    pub fn from_vec<V: Into<Validity>>(data: Vec<bool>, validity: V) -> Self {
        let validity = validity.into();
        // SAFETY: we are transmuting a Vec<bool> into a Vec<u8>
        let data: Vec<u8> = unsafe { std::mem::transmute(data) };
        Self::new(BufferHandle::new_host(ByteBuffer::from(data)), validity)
    }

    pub fn buffer(&self) -> &BufferHandle {
        &self.buffer
    }

    pub fn as_slice(&self) -> &[bool] {
        // Safety: The internal buffer contains byte-sized bools
        unsafe { std::mem::transmute(self.buffer().as_host().as_slice()) }
    }
}

impl ValidityHelper for ByteBoolArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

impl OperationsVTable<ByteBool> for ByteBool {
    fn scalar_at(
        array: &ByteBoolArray,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::bool(
            array.buffer.as_host()[index] == 1,
            array.dtype().nullability(),
        ))
    }
}

impl From<Vec<bool>> for ByteBoolArray {
    fn from(value: Vec<bool>) -> Self {
        Self::from_vec(value, Validity::AllValid)
    }
}

impl From<Vec<Option<bool>>> for ByteBoolArray {
    fn from(value: Vec<Option<bool>>) -> Self {
        let validity = Validity::from_iter(value.iter().map(|v| v.is_some()));

        // This doesn't reallocate, and the compiler even vectorizes it
        let data = value.into_iter().map(Option::unwrap_or_default).collect();

        Self::from_vec(data, validity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validity_construction() {
        let v = vec![true, false];
        let v_len = v.len();

        let arr = ByteBoolArray::from(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(arr.is_valid(idx).unwrap());
        }

        let v = vec![Some(true), None, Some(false)];
        let arr = ByteBoolArray::from(v);
        assert!(arr.is_valid(0).unwrap());
        assert!(!arr.is_valid(1).unwrap());
        assert!(arr.is_valid(2).unwrap());
        assert_eq!(arr.len(), 3);

        let v: Vec<Option<bool>> = vec![None, None];
        let v_len = v.len();

        let arr = ByteBoolArray::from(v);
        assert_eq!(v_len, arr.len());

        for idx in 0..arr.len() {
            assert!(!arr.is_valid(idx).unwrap());
        }
        assert_eq!(arr.len(), 2);
    }
}
