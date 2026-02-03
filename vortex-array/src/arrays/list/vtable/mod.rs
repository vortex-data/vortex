// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::ProstMetadata;
use crate::arrays::ListArray;
use crate::arrays::list::vtable::kernel::PARENT_KERNELS;
use crate::arrays::list_view_from_list;
use crate::buffer::BufferHandle;
use crate::metadata::DeserializeMetadata;
use crate::metadata::SerializeMetadata;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::NotSupported;
use crate::vtable::VTable;
use crate::vtable::ValidityHelper;
use crate::vtable::ValidityVTableFromValidityHelper;

mod array;
mod kernel;
mod operations;
mod validity;
mod visitor;

vtable!(List);

#[derive(Clone, prost::Message)]
pub struct ListMetadata {
    #[prost(uint64, tag = "1")]
    elements_len: u64,
    #[prost(enumeration = "PType", tag = "2")]
    offset_ptype: i32,
}

impl VTable for ListVTable {
    type Array = ListArray;

    type Metadata = ProstMetadata<ListMetadata>;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ListArray::new(
                array.elements().clone(),
                array.offsets().slice(range.start..range.end + 1)?,
                array.validity().slice(range)?,
            )
            .into_array(),
        ))
    }

    fn metadata(array: &ListArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(ListMetadata {
            elements_len: array.elements().len() as u64,
            offset_ptype: PType::try_from(array.offsets().dtype())? as i32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(SerializeMetadata::serialize(metadata)))
    }

    fn deserialize(bytes: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(
            <ProstMetadata<ListMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ListArray> {
        let validity = if children.len() == 2 {
            Validity::from(dtype.nullability())
        } else if children.len() == 3 {
            let validity = children.get(2, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 2 or 3 children, got {}", children.len());
        };

        let DType::List(element_dtype, _) = &dtype else {
            vortex_bail!("Expected List dtype, got {:?}", dtype);
        };
        let elements = children.get(
            0,
            element_dtype.as_ref(),
            usize::try_from(metadata.0.elements_len)?,
        )?;

        let offsets = children.get(
            1,
            &DType::Primitive(metadata.0.offset_ptype(), Nullability::NonNullable),
            len + 1,
        )?;

        ListArray::try_new(elements, offsets, validity)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 2 || children.len() == 3,
            "ListArray expects 2 or 3 children, got {}",
            children.len()
        );

        let mut iter = children.into_iter();
        let elements = iter
            .next()
            .vortex_expect("children length already validated");
        let offsets = iter
            .next()
            .vortex_expect("children length already validated");
        let validity = if let Some(validity_array) = iter.next() {
            Validity::Array(validity_array)
        } else {
            Validity::from(array.dtype.nullability())
        };

        let new_array = ListArray::try_new(elements, offsets, validity)?;
        *array = new_array;
        Ok(())
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
        Ok(Canonical::List(list_view_from_list(array.clone(), ctx)?))
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
pub struct ListVTable;

impl ListVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.list");
}
