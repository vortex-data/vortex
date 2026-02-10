// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_scalar::Scalar;
use vortex_scalar::ScalarValue;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::DeserializeMetadata;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::arrays::ConstantArray;
use crate::arrays::constant::ConstantMetadata;
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
const CONSTANT_INLINE_THRESHOLD: usize = 1024;

impl VTable for ConstantVTable {
    type Array = ConstantArray;

    type Metadata = ProstMetadata<ConstantMetadata>;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn metadata(array: &ConstantArray) -> VortexResult<Self::Metadata> {
        let proto_bytes: Vec<u8> = array.scalar().value().to_protobytes();
        let scalar_value = (proto_bytes.len() <= CONSTANT_INLINE_THRESHOLD).then_some(proto_bytes);
        Ok(ProstMetadata(ConstantMetadata { scalar_value }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        // Empty bytes indicates an old writer that didn't produce metadata.
        if bytes.is_empty() {
            return Ok(ProstMetadata(ConstantMetadata { scalar_value: None }));
        }
        let metadata = <Self::Metadata as DeserializeMetadata>::deserialize(bytes)?;
        Ok(ProstMetadata(metadata))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<ConstantArray> {
        // Prefer reading the scalar from inlined metadata to avoid device-to-host copies.
        let sv = if let Some(ref proto_bytes) = metadata.scalar_value {
            ScalarValue::from_protobytes(proto_bytes)?
        } else {
            if buffers.len() != 1 {
                vortex_bail!("Expected 1 buffer, got {}", buffers.len());
            }
            let buffer = buffers[0].clone().try_to_host_sync()?;
            ScalarValue::from_protobytes(&buffer)?
        };
        let scalar = Scalar::new(dtype.clone(), sv);
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
