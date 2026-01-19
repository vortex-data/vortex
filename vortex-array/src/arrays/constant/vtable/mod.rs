// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::ops::Range;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_scalar::Scalar;
use vortex_scalar::ScalarValue;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::constant::vtable::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::serde::ArrayChildren;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;

mod array;
mod canonical;
mod encode;
mod operations;
mod rules;
mod validity;
mod visitor;

vtable!(Constant);

#[derive(Debug)]
pub struct ConstantVTable;

impl VTable for ConstantVTable {
    type Array = ConstantArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    // TODO(ngates): implement a compute kernel for elementwise operations
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.constant")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        ConstantVTable.as_vtable()
    }

    fn metadata(_array: &ConstantArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(_buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<ConstantArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let buffer = buffers[0].clone().try_to_host()?;
        let sv = ScalarValue::from_protobytes(&buffer)?;
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

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ConstantArray::new(array.scalar.clone(), range.len()).into_array(),
        ))
    }
}
