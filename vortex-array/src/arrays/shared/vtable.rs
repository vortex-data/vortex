// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;
use vortex_session::VortexSession;

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::shared::SharedArray;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::BaseArrayVTable;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;
use crate::vtable::VisitorVTable;

vtable!(Shared);

// TODO(ngates): consider hooking Shared into the iterative execution model. Cache either the
//  most executed, or after each iteration, and return a shared cache for each execution.
#[derive(Debug)]
pub struct SharedVTable;

impl SharedVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.shared");
}

impl VTable for SharedVTable {
    type Array = SharedArray;
    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn metadata(_array: &Self::Array) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        vortex_error::vortex_bail!("Shared array is not serializable")
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        vortex_error::vortex_bail!("Shared array is not serializable")
    }

    fn build(
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[crate::buffer::BufferHandle],
        children: &dyn crate::serde::ArrayChildren,
    ) -> VortexResult<SharedArray> {
        let child = children.get(0, dtype, len)?;
        Ok(SharedArray::new(child))
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_error::vortex_ensure!(
            children.len() == 1,
            "SharedArray expects exactly 1 child, got {}",
            children.len()
        );
        let child = children
            .into_iter()
            .next()
            .vortex_expect("children length already validated");
        array.set_source(child);
        Ok(())
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(array
            .get_or_compute(|source| source.clone().execute::<Canonical>(ctx))?
            .into_array())
    }
}

impl BaseArrayVTable<SharedVTable> for SharedVTable {
    fn len(array: &SharedArray) -> usize {
        array.current_array_ref().len()
    }

    fn dtype(array: &SharedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &SharedArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &SharedArray, state: &mut H, precision: Precision) {
        let current = array.current_array_ref();
        current.array_hash(state, precision);
        array.dtype.hash(state);
    }

    fn array_eq(array: &SharedArray, other: &SharedArray, precision: Precision) -> bool {
        let current = array.current_array_ref();
        let other_current = other.current_array_ref();
        current.array_eq(&other_current, precision) && array.dtype == other.dtype
    }
}

impl OperationsVTable<SharedVTable> for SharedVTable {
    fn scalar_at(array: &SharedArray, index: usize) -> VortexResult<Scalar> {
        array.current_array_ref().scalar_at(index)
    }
}

impl ValidityVTable<SharedVTable> for SharedVTable {
    fn validity(array: &SharedArray) -> VortexResult<Validity> {
        array.current_array_ref().validity()
    }
}

impl VisitorVTable<SharedVTable> for SharedVTable {
    fn visit_buffers(_array: &SharedArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &SharedArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("source", &array.current_array_ref());
    }

    fn nchildren(_array: &SharedArray) -> usize {
        1
    }

    fn nth_child(array: &SharedArray, idx: usize) -> Option<ArrayRef> {
        match idx {
            0 => Some(array.current_array_ref()),
            _ => None,
        }
    }
}
