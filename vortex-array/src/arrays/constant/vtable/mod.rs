// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::ConstantArray;
use crate::arrays::constant::compute::rules::PARENT_RULES;
use crate::arrays::constant::vtable::canonical::constant_canonicalize;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
pub(crate) mod canonical;
mod operations;
mod validity;

vtable!(Constant);

#[derive(Debug)]
pub struct ConstantVTable;

impl ConstantVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.constant");
}

impl VTable for ConstantVTable {
    type Array = ConstantArray;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn len(array: &ConstantArray) -> usize {
        array.len
    }

    fn dtype(array: &ConstantArray) -> &DType {
        array.scalar.dtype()
    }

    fn stats(array: &ConstantArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &ConstantArray,
        state: &mut H,
        _precision: Precision,
    ) {
        array.scalar.hash(state);
        array.len.hash(state);
    }

    fn array_eq(array: &ConstantArray, other: &ConstantArray, _precision: Precision) -> bool {
        array.scalar == other.scalar && array.len == other.len
    }

    fn nbuffers(_array: &ConstantArray) -> usize {
        1
    }

    fn buffer(array: &ConstantArray, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(
                ScalarValue::to_proto_bytes::<ByteBufferMut>(array.scalar.value()).freeze(),
            ),
            _ => vortex_panic!("ConstantArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: &ConstantArray, idx: usize) -> Option<String> {
        match idx {
            0 => Some("scalar".to_string()),
            _ => None,
        }
    }

    fn nchildren(_array: &ConstantArray) -> usize {
        0
    }

    fn child(_array: &ConstantArray, idx: usize) -> ArrayRef {
        vortex_panic!("ConstantArray child index {idx} out of bounds")
    }

    fn child_name(_array: &ConstantArray, idx: usize) -> String {
        vortex_panic!("ConstantArray child_name index {idx} out of bounds")
    }

    fn metadata(_array: &ConstantArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(Vec::new()))
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
        _children: &dyn ArrayChildren,
    ) -> VortexResult<ConstantArray> {
        vortex_ensure!(
            buffers.len() == 1,
            "Expected 1 buffer, got {}",
            buffers.len()
        );

        let buffer = buffers[0].clone().try_to_host_sync()?;
        let bytes: &[u8] = buffer.as_ref();

        let scalar_value = ScalarValue::from_proto_bytes(bytes, dtype)?;
        let scalar = Scalar::try_new(dtype.clone(), scalar_value)?;

        Ok(ConstantArray::new(scalar, len))
    }

    fn with_children(_array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.is_empty(),
            "ConstantArray has no children, got {}",
            children.len()
        );
        Ok(())
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(constant_canonicalize(array)?.into_array())
    }
}
