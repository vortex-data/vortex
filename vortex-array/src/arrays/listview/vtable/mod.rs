// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_vector::Vector;
use vortex_vector::listview::ListViewVector;

use crate::ArrayOperator;
use crate::DeserializeMetadata;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::arrays::ListViewArray;
use crate::execution::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;

mod array;
mod canonical;
mod operations;
mod operator;
mod validity;
mod visitor;

vtable!(ListView);

#[derive(Debug)]
pub struct ListViewVTable;

#[derive(Clone, prost::Message)]
pub struct ListViewMetadata {
    #[prost(uint64, tag = "1")]
    elements_len: u64,
    #[prost(enumeration = "PType", tag = "2")]
    offset_ptype: i32,
    #[prost(enumeration = "PType", tag = "3")]
    size_ptype: i32,
}

impl VTable for ListViewVTable {
    type Array = ListViewArray;

    type Metadata = ProstMetadata<ListViewMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type OperatorVTable = Self;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.listview")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        ListViewVTable.as_vtable()
    }

    fn metadata(array: &ListViewArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(ListViewMetadata {
            elements_len: array.elements().len() as u64,
            offset_ptype: PType::try_from(array.offsets().dtype())? as i32,
            size_ptype: PType::try_from(array.sizes().dtype())? as i32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(bytes: &[u8]) -> VortexResult<Self::Metadata> {
        let metadata = <Self::Metadata as DeserializeMetadata>::deserialize(bytes)?;
        Ok(ProstMetadata(metadata))
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ListViewArray> {
        vortex_ensure!(
            buffers.is_empty(),
            "`ListViewArray::build` expects no buffers"
        );

        let DType::List(element_dtype, _) = dtype else {
            vortex_bail!("Expected List dtype, got {:?}", dtype);
        };

        let validity = if children.len() == 3 {
            Validity::from(dtype.nullability())
        } else if children.len() == 4 {
            let validity = children.get(3, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!(
                "`ListViewArray::build` expects 3 or 4 children, got {}",
                children.len()
            );
        };

        // Get elements with the correct length from metadata.
        let elements = children.get(
            0,
            element_dtype.as_ref(),
            usize::try_from(metadata.0.elements_len)?,
        )?;

        // Get offsets with proper type from metadata.
        let offsets = children.get(
            1,
            &DType::Primitive(metadata.0.offset_ptype(), Nullability::NonNullable),
            len,
        )?;

        // Get sizes with proper type from metadata.
        let sizes = children.get(
            2,
            &DType::Primitive(metadata.0.size_ptype(), Nullability::NonNullable),
            len,
        )?;

        ListViewArray::try_new(elements, offsets, sizes, validity)
    }

    fn execute(array: &Self::Array, ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        Ok(unsafe {
            ListViewVector::new_unchecked(
                Arc::new(array.elements().execute_batch(ctx)?),
                array.offsets().execute_batch(ctx)?.into_primitive(),
                array.sizes().execute_batch(ctx)?.into_primitive(),
                array.validity_mask(),
            )
        }
        .into())
    }
}
