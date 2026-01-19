// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
mod canonical;
mod operations;
mod rules;
mod validity;
mod visitor;

use std::ops::Range;

use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::IntoArray;
use crate::arrays::extension::ExtensionArray;
use crate::arrays::extension::vtable::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::serde::ArrayChildren;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromChild;

vtable!(Extension);

impl VTable for ExtensionVTable {
    type Array = ExtensionArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.ext")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        ExtensionVTable.as_vtable()
    }

    fn metadata(_array: &ExtensionArray) -> VortexResult<Self::Metadata> {
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
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ExtensionArray> {
        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("Not an extension DType");
        };
        if children.len() != 1 {
            vortex_bail!("Expected 1 child, got {}", children.len());
        }
        let storage = children.get(0, ext_dtype.storage_dtype(), len)?;
        Ok(ExtensionArray::new(ext_dtype.clone(), storage))
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1,
            "ExtensionArray expects exactly 1 child (storage), got {}",
            children.len()
        );
        array.storage = children
            .into_iter()
            .next()
            .vortex_expect("children length already validated");
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
            ExtensionArray::new(array.ext_dtype().clone(), array.storage().slice(range))
                .into_array(),
        ))
    }
}

#[derive(Debug)]
pub struct ExtensionVTable;
