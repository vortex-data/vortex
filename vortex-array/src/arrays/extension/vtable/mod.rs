// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
mod canonical;
mod kernel;
mod operations;
mod validity;

use std::hash::Hash;
use std::sync::Arc;

use kernel::PARENT_KERNELS;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::Precision;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::array::NUM_SLOTS;
use crate::arrays::extension::array::SLOT_NAMES;
use crate::arrays::extension::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::Array;
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
        array.storage_array().len()
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
        array.storage_array().array_hash(state, precision);
    }

    fn array_eq(array: &ExtensionArray, other: &ExtensionArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array
                .storage_array()
                .array_eq(other.storage_array(), precision)
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

    fn slots(array: &ExtensionArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &ExtensionArray, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
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

    fn with_slots(array: &mut Self::Array, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "ExtensionArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

#[derive(Clone, Debug)]
pub struct Extension;

impl Extension {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.ext");
}
