// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_scalar::Scalar;
use vortex_scalar::ScalarValue;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::constant::compute::rules::PARENT_RULES;
use crate::arrays::constant::vtable::canonical::constant_canonicalize;
use crate::buffer::BufferHandle;
use crate::serde::ArrayChildren;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;

mod array;
pub(crate) mod canonical;
mod operations;
mod validity;
mod visitor;

vtable!(Constant);

#[derive(Debug)]
pub struct ConstantVTable;

impl ConstantVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.constant");
}

/// Maximum size (in bytes) of a protobuf-encoded scalar value that will be inlined
/// into the array metadata. Values larger than this are stored only in the buffer.
pub(crate) const CONSTANT_INLINE_THRESHOLD: usize = 1024;

impl VTable for ConstantVTable {
    type Array = ConstantArray;

    /// The scalar constant value.
    ///
    /// During serialization, scalars small enough (<= `CONSTANT_INLINE_THRESHOLD` bytes) are
    /// inlined into the metadata bytes. Larger scalars are stored only in the buffer and
    /// reconstructed from it during deserialization.
    type Metadata = Scalar;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn metadata(array: &ConstantArray) -> VortexResult<Self::Metadata> {
        Ok(array.scalar().clone())
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        // If the scalar is small enough, inline it into the metadata bytes.
        // Note that we **only** serialize the scalar value (not including the dtype).
        Ok(Some(if metadata.nbytes() <= CONSTANT_INLINE_THRESHOLD {
            ScalarValue::to_proto_bytes(metadata.value())
        } else {
            // Large scalars are stored only in the buffer; return empty bytes.
            Vec::new()
        }))
    }

    fn deserialize(
        bytes: &[u8],
        dtype: &DType,
        _len: usize,
        buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        if !bytes.is_empty() {
            // If metadata has been deserialized, then it means we can fast-path read the constant
            // directly from the metadata bytes.
            let scalar_value = ScalarValue::from_proto_bytes(bytes, dtype)?;
            return Scalar::try_new(dtype.clone(), scalar_value);
        }

        vortex_ensure!(
            buffers.len() == 1,
            "Expected 1 buffer for the constant scalar, got {}",
            buffers.len()
        );

        // Otherwise, the scalar was too large to inline / serialize into metadata, so we
        // reconstruct it now from the buffer.

        let buffer = buffers[0].clone().try_to_host_sync()?;
        let bytes: &[u8] = buffer.as_ref();

        let scalar_value = ScalarValue::from_proto_bytes(bytes, dtype)?;
        Scalar::try_new(dtype.clone(), scalar_value)
    }

    fn build(
        _dtype: &DType,
        len: usize,
        metadata: &Scalar,
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<ConstantArray> {
        Ok(ConstantArray::new(metadata.clone(), len))
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
