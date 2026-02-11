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

    /// Optional inlined scalar constant.
    ///
    /// When the scalar value is small enough (<= `CONSTANT_INLINE_THRESHOLD` bytes), it is stored
    /// directly in the metadata to avoid an extra buffer allocation and potential
    /// device-to-host copy during deserialization.
    ///
    /// Currently, scalars are **always** stored in a separate buffer, regardless of if we inline a
    /// small scalar into the metadata.
    type Metadata = Option<Scalar>;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn metadata(array: &ConstantArray) -> VortexResult<Self::Metadata> {
        let constant = array.scalar();

        // If the scalar is small enough, we can simply carry it around as metadata.
        Ok((constant.nbytes() <= CONSTANT_INLINE_THRESHOLD).then_some(constant.clone()))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        // If we do not have a scalar to serialize, just return empty bytes.
        Ok(Some(metadata.map_or_else(Vec::new, |c| {
            // Note that we **only** serialize the optional scalar value (not including the dtype).
            ScalarValue::to_proto_bytes(c.value())
        })))
    }

    fn deserialize(
        bytes: &[u8],
        dtype: &DType,
        _len: usize,
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        // Empty bytes indicates an old writer that didn't produce metadata.
        if bytes.is_empty() {
            return Ok(None);
        }

        // Otherwise, deserialize the constant scalar from the metadata.
        let scalar_value = ScalarValue::from_proto_bytes(bytes, dtype)?;
        Some(Scalar::try_new(dtype.clone(), scalar_value)).transpose()
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<ConstantArray> {
        // Prefer reading the scalar from inlined metadata to avoid device-to-host copies.
        if let Some(constant) = metadata {
            return Ok(ConstantArray::new(constant.clone(), len));
        }

        // Otherwise, get the constant scalar from the buffers.
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
