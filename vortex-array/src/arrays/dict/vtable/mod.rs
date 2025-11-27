// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::DictArray;
use super::DictMetadata;
use crate::DeserializeMetadata;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::serde::ArrayChildren;
use crate::vtable;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;

mod array;
mod canonical;
mod encode;
mod operations;
mod validity;
mod visitor;

vtable!(Dict);

#[derive(Debug)]
pub struct DictVTable;

impl VTable for DictVTable {
    type Array = DictArray;

    type Metadata = ProstMetadata<DictMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type OperatorVTable = NotSupported;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.dict")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        DictVTable.as_vtable()
    }

    fn metadata(array: &DictArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DictMetadata {
            codes_ptype: PType::try_from(array.codes().dtype())? as i32,
            values_len: u32::try_from(array.values().len()).map_err(|_| {
                vortex_err!(
                    "Dictionary values size {} overflowed u32",
                    array.values().len()
                )
            })?,
            is_nullable_codes: Some(array.codes().dtype().is_nullable()),
            all_values_referenced: Some(array.all_values_referenced),
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        let metadata = <Self::Metadata as DeserializeMetadata>::deserialize(buffer)?;
        Ok(ProstMetadata(metadata))
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DictArray> {
        if children.len() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                children.len()
            )
        }
        let codes_nullable = metadata
            .is_nullable_codes
            .map(Nullability::from)
            // If no `is_nullable_codes` metadata use the nullability of the values
            // (and whole array) as before.
            .unwrap_or_else(|| dtype.nullability());
        let codes_dtype = DType::Primitive(metadata.codes_ptype(), codes_nullable);
        let codes = children.get(0, &codes_dtype, len)?;
        let values = children.get(1, dtype, metadata.values_len as usize)?;
        let all_values_referenced = metadata.all_values_referenced.unwrap_or(false);

        // SAFETY: We've validated the metadata and children.
        Ok(unsafe {
            DictArray::new_unchecked(codes, values).set_all_values_referenced(all_values_referenced)
        })
    }
}
