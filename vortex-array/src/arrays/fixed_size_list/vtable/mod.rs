// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

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
use crate::arrays::FixedSizeListArray;
use crate::arrays::fixed_size_list::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;
use crate::vtable::validity_nchildren;
use crate::vtable::validity_to_child;
mod kernel;
mod operations;
mod validity;
vtable!(FixedSizeList);

#[derive(Debug)]
pub struct FixedSizeList;

impl FixedSizeList {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.fixed_size_list");
}

impl VTable for FixedSizeList {
    type Array = FixedSizeListArray;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    fn vtable(_array: &Self::Array) -> &Self {
        &FixedSizeList
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &FixedSizeListArray) -> usize {
        array.len
    }

    fn dtype(array: &FixedSizeListArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &FixedSizeListArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &FixedSizeListArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.elements().array_hash(state, precision);
        array.list_size().hash(state);
        array.validity.array_hash(state, precision);
        array.len.hash(state);
    }

    fn array_eq(
        array: &FixedSizeListArray,
        other: &FixedSizeListArray,
        precision: Precision,
    ) -> bool {
        array.dtype == other.dtype
            && array.elements().array_eq(other.elements(), precision)
            && array.list_size() == other.list_size()
            && array.validity.array_eq(&other.validity, precision)
            && array.len == other.len
    }

    fn nbuffers(_array: &FixedSizeListArray) -> usize {
        0
    }

    fn buffer(_array: &FixedSizeListArray, idx: usize) -> BufferHandle {
        vortex_panic!("FixedSizeListArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &FixedSizeListArray, idx: usize) -> Option<String> {
        vortex_panic!("FixedSizeListArray buffer_name index {idx} out of bounds")
    }

    fn nchildren(array: &FixedSizeListArray) -> usize {
        1 + validity_nchildren(&array.validity)
    }

    fn child(array: &FixedSizeListArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.elements().clone(),
            1 => validity_to_child(&array.validity, array.len())
                .vortex_expect("FixedSizeListArray validity child out of bounds"),
            _ => vortex_panic!("FixedSizeListArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &FixedSizeListArray, idx: usize) -> String {
        match idx {
            0 => "elements".to_string(),
            1 => "validity".to_string(),
            _ => vortex_panic!("FixedSizeListArray child_name index {idx} out of bounds"),
        }
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
        Self::PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn metadata(_array: &FixedSizeListArray) -> VortexResult<Self::Metadata> {
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

    /// Builds a [`FixedSizeListArray`].
    ///
    /// This method expects 1 or 2 children (a second child indicates a validity array).
    fn build(
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FixedSizeListArray> {
        vortex_ensure!(
            buffers.is_empty(),
            "`FixedSizeList::build` expects no buffers"
        );

        let DType::FixedSizeList(element_dtype, list_size, _) = &dtype else {
            vortex_bail!("Expected `DType::FixedSizeList`, got {:?}", dtype);
        };

        let validity = {
            if children.len() > 2 {
                vortex_bail!("`FixedSizeList::build` method expected 1 or 2 children")
            }

            if children.len() == 2 {
                let validity = children.get(1, &Validity::DTYPE, len)?;
                Validity::Array(validity)
            } else {
                debug_assert_eq!(children.len(), 1);
                Validity::from(dtype.nullability())
            }
        };

        let num_elements = len * (*list_size as usize);
        let elements = children.get(0, element_dtype.as_ref(), num_elements)?;

        FixedSizeListArray::try_new(elements, *list_size, validity, len)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 1 || children.len() == 2,
            "FixedSizeListArray expects 1 or 2 children, got {}",
            children.len()
        );

        let mut iter = children.into_iter();
        let elements = iter
            .next()
            .vortex_expect("children length already validated");
        let validity = if let Some(validity_array) = iter.next() {
            Validity::Array(validity_array)
        } else {
            Validity::from(array.dtype.nullability())
        };

        let new_array =
            FixedSizeListArray::try_new(elements, array.list_size(), validity, array.len())?;
        *array = new_array;
        Ok(())
    }

    fn execute(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        Ok(ExecutionStep::Done(array.clone().into_array()))
    }
}
