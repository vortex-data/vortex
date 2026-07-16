// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::VTable;
use crate::array::with_empty_buffers;
use crate::arrays::union::UnionArrayExt;
use crate::arrays::union::array::CHILDREN_OFFSET;
use crate::arrays::union::array::TYPE_IDS_SLOT;
use crate::arrays::union::array::make_union_parts;
use crate::arrays::union::compute::rules::PARENT_RULES;
use crate::arrays::union::type_ids_dtype;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::serde::ArrayChildren;

mod operations;
mod validity;

/// A canonical sparse [`Union`]-encoded array.
pub type UnionArray = Array<Union>;

/// The canonical sparse Union encoding.
#[derive(Clone, Debug)]
pub struct Union;

impl VTable for Union {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.union");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Union(variants, nullability) = dtype else {
            vortex_bail!("Expected union dtype, found {dtype}")
        };
        vortex_ensure!(
            slots.len() == CHILDREN_OFFSET + variants.len(),
            "UnionArray has {} slots but expected {}",
            slots.len(),
            CHILDREN_OFFSET + variants.len()
        );

        let type_ids = slots[TYPE_IDS_SLOT]
            .as_ref()
            .ok_or_else(|| vortex_err!("UnionArray is missing its type_ids slot"))?;
        let expected_type_ids_dtype = type_ids_dtype(*nullability);
        vortex_ensure!(
            type_ids.dtype() == &expected_type_ids_dtype,
            "UnionArray type_ids must have dtype {expected_type_ids_dtype}, got {}",
            type_ids.dtype()
        );
        vortex_ensure!(
            type_ids.len() == len,
            "UnionArray type_ids length {} does not match outer length {len}",
            type_ids.len()
        );

        for (index, variant_dtype) in variants.variants().enumerate() {
            let child = slots[CHILDREN_OFFSET + index]
                .as_ref()
                .ok_or_else(|| vortex_err!("UnionArray is missing child {index}"))?;
            vortex_ensure!(
                child.len() == len,
                "UnionArray child {index} length {} does not match outer length {len}",
                child.len()
            );
            vortex_ensure!(
                child.dtype() == &variant_dtype,
                "UnionArray child {index} has dtype {} but expected {variant_dtype}",
                child.dtype()
            );
        }

        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("UnionArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("UnionArray buffer_name index {idx} out of bounds")
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(metadata.is_empty(), "UnionArray expects empty metadata");
        vortex_ensure!(buffers.is_empty(), "UnionArray expects no buffers");
        let DType::Union(variants, nullability) = dtype else {
            vortex_bail!("Expected union dtype, found {dtype}")
        };
        vortex_ensure!(
            children.len() == CHILDREN_OFFSET + variants.len(),
            "UnionArray expected {} children, found {}",
            CHILDREN_OFFSET + variants.len(),
            children.len()
        );

        let type_ids = children.get(TYPE_IDS_SLOT, &type_ids_dtype(*nullability), len)?;
        let sparse_children = variants
            .variants()
            .enumerate()
            .map(|(index, dtype)| children.get(CHILDREN_OFFSET + index, &dtype, len))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(make_union_parts(
            type_ids,
            variants.clone(),
            &sparse_children,
        ))
    }

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        if idx == TYPE_IDS_SLOT {
            "type_ids".to_string()
        } else {
            array.variants().names()[idx - CHILDREN_OFFSET].to_string()
        }
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn append_to_builder(
        _array: ArrayView<'_, Self>,
        _builder: &mut dyn ArrayBuilder,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_bail!("append_to_builder is not supported for UnionArray yet")
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}
