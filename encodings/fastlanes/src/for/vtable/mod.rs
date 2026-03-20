// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionStep;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::FoRArray;
use crate::r#for::array::for_decompress::decompress;
use crate::r#for::vtable::kernels::PARENT_KERNELS;
use crate::r#for::vtable::rules::PARENT_RULES;

mod kernels;
mod operations;
mod rules;
mod slice;
mod validity;

vtable!(FoR);

impl VTable for FoR {
    type Array = FoRArray;

    type Metadata = Scalar;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &FoR
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &FoRArray) -> usize {
        array.encoded().len()
    }

    fn dtype(array: &FoRArray) -> &DType {
        array.reference_scalar().dtype()
    }

    fn stats(array: &FoRArray) -> StatsSetRef<'_> {
        array.stats_set().to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &FoRArray, state: &mut H, precision: Precision) {
        array.encoded().array_hash(state, precision);
        array.reference_scalar().hash(state);
    }

    fn array_eq(array: &FoRArray, other: &FoRArray, precision: Precision) -> bool {
        array.encoded().array_eq(other.encoded(), precision)
            && array.reference_scalar() == other.reference_scalar()
    }

    fn nbuffers(_array: &FoRArray) -> usize {
        0
    }

    fn buffer(_array: &FoRArray, idx: usize) -> BufferHandle {
        vortex_panic!("FoRArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &FoRArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(_array: &FoRArray) -> usize {
        1
    }

    fn child(array: &FoRArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.encoded().clone(),
            _ => vortex_panic!("FoRArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &FoRArray, idx: usize) -> String {
        match idx {
            0 => "encoded".to_string(),
            _ => vortex_panic!("FoRArray child name index {idx} out of bounds"),
        }
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        // FoRArray children order (from visit_children):
        // 1. encoded

        vortex_ensure!(
            children.len() == 1,
            "Expected 1 child for FoR encoding, got {}",
            children.len()
        );

        array.encoded = children[0].clone();

        Ok(())
    }

    fn metadata(array: &FoRArray) -> VortexResult<Self::Metadata> {
        Ok(array.reference_scalar().clone())
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        // Note that we **only** serialize the optional scalar value (not including the dtype).
        Ok(Some(ScalarValue::to_proto_bytes(metadata.value())))
    }

    fn deserialize(
        bytes: &[u8],
        dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let scalar_value = ScalarValue::from_proto_bytes(bytes, dtype, session)?;
        Scalar::try_new(dtype.clone(), scalar_value)
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FoRArray> {
        if children.len() != 1 {
            vortex_bail!(
                "Expected 1 child for FoR encoding, found {}",
                children.len()
            )
        }

        let encoded = children.get(0, dtype, len)?;

        FoRArray::try_new(encoded, metadata.clone())
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        Ok(ExecutionStep::Done(decompress(array, ctx)?.into_array()))
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
pub struct FoR;

impl FoR {
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.for");
}
