// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
mod canonical;
mod kernel;
mod operations;
mod validity;

use std::hash::Hash;

use kernel::PARENT_KERNELS;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionStep;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromChild;

vtable!(Extension);

impl VTable for Extension {
    type Array = ExtensionArray;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &Extension
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ExtensionArray) -> usize {
        array.storage_array.len()
    }

    fn dtype(array: &ExtensionArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ExtensionArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &ExtensionArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.storage_array.array_hash(state, precision);
    }

    fn array_eq(array: &ExtensionArray, other: &ExtensionArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array
                .storage_array
                .array_eq(&other.storage_array, precision)
    }

    fn nbuffers(_array: &ExtensionArray) -> usize {
        0
    }

    fn buffer(_array: &ExtensionArray, idx: usize) -> BufferHandle {
        vortex_panic!("ExtensionArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &ExtensionArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &ExtensionArray) -> usize {
        1
    }

    fn child(array: &ExtensionArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.storage_array.clone(),
            _ => vortex_panic!("ExtensionArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &ExtensionArray, idx: usize) -> String {
        match idx {
            0 => "storage".to_string(),
            _ => vortex_panic!("ExtensionArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(_array: &ExtensionArray) -> VortexResult<Self::Metadata> {
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
        array.storage_array = children
            .into_iter()
            .next()
            .vortex_expect("children length already validated");
        Ok(())
    }

    fn execute(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        Ok(ExecutionStep::Done(array.clone().into_array()))
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
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

#[derive(Debug)]
pub struct Extension;

impl Extension {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.ext");
}
