// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::arrays::FixedSizeListArray;
use crate::arrays::fixed_size_list::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;

mod array;
mod operations;
mod validity;
mod visitor;

vtable!(FixedSizeList);

#[derive(Debug)]
pub struct FixedSizeListVTable;

impl FixedSizeListVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.fixed_size_list");
}

impl VTable for FixedSizeListVTable {
    type Array = FixedSizeListArray;

    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
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
            "`FixedSizeListVTable::build` expects no buffers"
        );

        let DType::FixedSizeList(element_dtype, list_size, _) = &dtype else {
            vortex_bail!("Expected `DType::FixedSizeList`, got {:?}", dtype);
        };

        let validity = {
            if children.len() > 2 {
                vortex_bail!("`FixedSizeListVTable::build` method expected 1 or 2 children")
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

    fn execute(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(array.to_array())
    }
}
