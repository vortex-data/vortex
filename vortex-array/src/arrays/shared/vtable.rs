// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::ops::Range;

use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

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
use crate::vtable::NotSupported;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;
use crate::vtable::VisitorVTable;

vtable!(Shared);

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
    type ComputeVTable = NotSupported;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn metadata(_array: &Self::Array) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        vortex_error::vortex_bail!("Shared array is not serializable")
    }

    fn deserialize(_bytes: &[u8]) -> VortexResult<Self::Metadata> {
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
        array.source = children
            .into_iter()
            .next()
            .vortex_expect("children length already validated");
        Ok(())
    }

    fn canonicalize(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        array.canonicalize(ctx)
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let sliced = array.source.slice(range)?;
        Ok(Some(SharedArray::new(sliced).into_array()))
    }
}

impl BaseArrayVTable<SharedVTable> for SharedVTable {
    fn len(array: &SharedArray) -> usize {
        array.source.len()
    }

    fn dtype(array: &SharedArray) -> &DType {
        array.source.dtype()
    }

    fn stats(array: &SharedArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &SharedArray, state: &mut H, precision: Precision) {
        array.source.array_hash(state, precision);
        array.source.dtype().hash(state);
    }

    fn array_eq(array: &SharedArray, other: &SharedArray, precision: Precision) -> bool {
        array.source.array_eq(&other.source, precision)
            && array.source.dtype() == other.source.dtype()
    }
}

impl OperationsVTable<SharedVTable> for SharedVTable {
    fn scalar_at(array: &SharedArray, index: usize) -> VortexResult<Scalar> {
        array.source.scalar_at(index)
    }
}

impl ValidityVTable<SharedVTable> for SharedVTable {
    fn validity(array: &SharedArray) -> VortexResult<Validity> {
        array.source.validity()
    }
}

impl VisitorVTable<SharedVTable> for SharedVTable {
    fn visit_buffers(_array: &SharedArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &SharedArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("source", array.source());
    }
}
