// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;
use std::sync::Arc;

use vortex::array::ArrayEq;
use vortex::array::ArrayHash;
use vortex::array::ArrayRef;
use vortex::array::EmptyMetadata;
use vortex::array::ExecutionCtx;
use vortex::array::ExecutionResult;
use vortex::array::Precision;
use vortex::array::buffer::BufferHandle;
use vortex::array::serde::ArrayChildren;
use vortex::array::stats::StatsSetRef;
use vortex::array::vtable;
use vortex::array::vtable::Array;
use vortex::array::vtable::ArrayId;
use vortex::array::vtable::VTable;
use vortex::array::vtable::ValidityVTableFromChild;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure_eq;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;
use vortex::session::VortexSession;

use crate::encodings::norm::array::NormVectorArray;
use crate::utils::extension_element_ptype;

mod operations;
mod validity;

pub(super) const VECTORS_SLOT: usize = 0;
pub(super) const NORMS_SLOT: usize = 1;
pub(super) const NUM_SLOTS: usize = 2;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["vectors", "norms"];

vtable!(NormVector);

#[derive(Debug, Clone)]
pub struct NormVector;

impl VTable for NormVector {
    type Array = NormVectorArray;
    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &NormVector
    }

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.tensor.norm_vector")
    }

    fn len(array: &NormVectorArray) -> usize {
        array.vector_array().len()
    }

    fn dtype(array: &NormVectorArray) -> &DType {
        array.vector_array().dtype()
    }

    fn stats(array: &NormVectorArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &NormVectorArray, state: &mut H, precision: Precision) {
        array.vector_array().array_hash(state, precision);
        array.norms().array_hash(state, precision);
    }

    fn array_eq(array: &NormVectorArray, other: &NormVectorArray, precision: Precision) -> bool {
        array.norms().array_eq(other.norms(), precision)
            && array
                .vector_array()
                .array_eq(other.vector_array(), precision)
    }

    fn nbuffers(_array: &NormVectorArray) -> usize {
        0
    }

    fn buffer(_array: &NormVectorArray, idx: usize) -> BufferHandle {
        vortex_panic!("NormVectorArray has no buffers (index {idx})")
    }

    fn buffer_name(_array: &NormVectorArray, idx: usize) -> Option<String> {
        vortex_panic!("NormVectorArray has no buffers (index {idx})")
    }

    fn nchildren(_array: &NormVectorArray) -> usize {
        2
    }

    fn child(array: &NormVectorArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.vector_array().clone(),
            1 => array.norms().clone(),
            _ => vortex_panic!("NormVectorArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &NormVectorArray, idx: usize) -> String {
        match idx {
            0 => "vector_array".to_string(),
            1 => "norms".to_string(),
            _ => vortex_panic!("NormVectorArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(_array: &NormVectorArray) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<NormVectorArray> {
        vortex_ensure_eq!(
            children.len(),
            2,
            "NormVectorArray requires exactly 2 children"
        );

        let vector_array = children.get(0, dtype, len)?;

        let ext = dtype.as_extension_opt().ok_or_else(|| {
            vortex_err!("NormVectorArray dtype must be an extension type, got {dtype}")
        })?;
        let element_ptype = extension_element_ptype(ext)?;
        let nullability = Nullability::from(dtype.is_nullable());
        let norms_dtype = DType::Primitive(element_ptype, nullability);
        let norms = children.get(1, &norms_dtype, len)?;

        NormVectorArray::try_new(vector_array, norms)
    }

    fn slots(array: &Self::Array) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &Self::Array, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::Array, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure_eq!(
            slots.len(),
            NUM_SLOTS,
            "FixedSizeListArray expects exactly {NUM_SLOTS} slots, got {}",
            slots.len()
        );

        array.slots = slots;
        Ok(())
    }

    fn execute(
        array: Arc<Array<NormVector>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array.decompress(ctx)?))
    }
}
