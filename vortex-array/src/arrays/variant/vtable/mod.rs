// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operations;
mod validity;

use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::Precision;
use crate::arrays::variant::VariantData;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayInner;
use crate::vtable::ArrayView;
use crate::vtable::VTable;

vtable!(Variant, Variant, VariantData);

#[derive(Clone, Debug)]
pub struct Variant;

impl Variant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.variant");
}

impl VTable for Variant {
    type ArrayData = VariantData;

    type Metadata = EmptyMetadata;

    type OperationsVTable = Self;

    type ValidityVTable = Self;

    fn vtable(_array: &Self::ArrayData) -> &Self {
        &Variant
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &Self::ArrayData) -> usize {
        array.child.len()
    }

    fn dtype(array: &Self::ArrayData) -> &DType {
        array.child.dtype()
    }

    fn stats(array: &Self::ArrayData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: Hasher>(array: ArrayView<'_, Self>, state: &mut H, precision: Precision) {
        array.child.array_hash(state, precision);
    }

    fn array_eq(
        array: ArrayView<'_, Self>,
        other: ArrayView<'_, Self>,
        precision: Precision,
    ) -> bool {
        array.child.array_eq(&other.child, precision)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("VariantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn child(array: ArrayView<'_, Self>, idx: usize) -> ArrayRef {
        match idx {
            0 => array.child.clone(),
            _ => vortex_panic!("VariantArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match idx {
            0 => "child".to_string(),
            _ => vortex_panic!("VariantArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(_array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<Self::ArrayData> {
        vortex_ensure!(matches!(dtype, DType::Variant(_)), "Expected Variant DType");
        vortex_ensure!(
            children.len() == 1,
            "Expected 1 child, got {}",
            children.len()
        );
        // The child carries the nullability for the whole VariantArray.
        let child = children.get(0, dtype, len)?;
        Ok(VariantData::new(child))
    }

    fn with_children(array: &mut Self::ArrayData, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1,
            "VariantArray expects exactly 1 child, got {}",
            children.len()
        );
        array.child = children
            .into_iter()
            .next()
            .vortex_expect("VariantArray must have 1 child");
        Ok(())
    }

    fn execute(
        array: Arc<ArrayInner<Self>>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }
}
