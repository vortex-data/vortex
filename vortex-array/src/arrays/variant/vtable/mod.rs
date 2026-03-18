// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operations;
mod validity;

use std::hash::Hasher;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionStep;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::VariantArray;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;

vtable!(Variant);

#[derive(Debug)]
pub struct Variant;

impl Variant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.variant");
}

impl VTable for Variant {
    type Array = VariantArray;

    type Metadata = EmptyMetadata;

    type OperationsVTable = Self;

    type ValidityVTable = Self;

    fn vtable(_array: &Self::Array) -> &Self {
        &Variant
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &Self::Array) -> usize {
        array.child.len()
    }

    fn dtype(array: &Self::Array) -> &DType {
        &array.dtype
    }

    fn stats(array: &Self::Array) -> StatsSetRef<'_> {
        array.child.statistics()
    }

    fn array_hash<H: Hasher>(array: &Self::Array, state: &mut H, precision: Precision) {
        array.child.array_hash(state, precision);
    }

    fn array_eq(array: &Self::Array, other: &Self::Array, precision: Precision) -> bool {
        array.child.array_eq(&other.child, precision)
    }

    fn nbuffers(_array: &Self::Array) -> usize {
        0
    }

    fn buffer(_array: &Self::Array, idx: usize) -> BufferHandle {
        vortex_panic!("VariantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &Self::Array, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &Self::Array) -> usize {
        1
    }

    fn child(array: &Self::Array, idx: usize) -> ArrayRef {
        match idx {
            0 => array.child.clone(),
            _ => vortex_panic!("VariantArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &Self::Array, idx: usize) -> String {
        match idx {
            0 => "child".to_string(),
            _ => vortex_panic!("VariantArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(_array: &Self::Array) -> VortexResult<Self::Metadata> {
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
        _session: &vortex_session::VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array> {
        vortex_ensure!(matches!(dtype, DType::Variant(_)), "Expected Variant DType");
        vortex_ensure!(
            children.len() == 1,
            "Expected 1 child, got {}",
            children.len()
        );
        // The child can be any variant encoding, so we use DType::Variant.
        let child = children.get(
            0,
            &DType::Variant(crate::dtype::Nullability::NonNullable),
            len,
        )?;
        Ok(VariantArray::new(child, dtype.nullability()))
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1,
            "VariantArray expects exactly 1 child, got {}",
            children.len()
        );
        array.child = children.into_iter().next().vortex_expect("must exist");
        Ok(())
    }

    fn execute(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        // VariantArray is the canonical variant representation.
        Ok(ExecutionStep::done(array.clone().into_array()))
    }
}
