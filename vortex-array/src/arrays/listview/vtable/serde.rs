// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexResult, vortex_bail, vortex_ensure};

use crate::arrays::{ListViewArray, ListViewEncoding, ListViewShape, ListViewVTable};
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;
use crate::{Array, ProstMetadata};

#[derive(Clone, prost::Message)]
pub struct ListViewMetadata {
    #[prost(uint64, tag = "1")]
    elements_len: u64,
    #[prost(enumeration = "PType", tag = "2")]
    offset_ptype: i32,
    #[prost(enumeration = "PType", tag = "3")]
    size_ptype: i32,
    #[prost(bool, tag = "4", default = false)]
    has_sorted_offsets: bool,
    #[prost(bool, tag = "5", default = false)]
    has_no_overlaps: bool,
    #[prost(bool, tag = "6", default = false)]
    has_no_gaps: bool,
}

impl SerdeVTable<ListViewVTable> for ListViewVTable {
    type Metadata = ProstMetadata<ListViewMetadata>;

    fn metadata(array: &ListViewArray) -> VortexResult<Option<Self::Metadata>> {
        let shape = array.shape();
        Ok(Some(ProstMetadata(ListViewMetadata {
            elements_len: array.elements().len() as u64,
            offset_ptype: PType::try_from(array.offsets().dtype())? as i32,
            size_ptype: PType::try_from(array.sizes().dtype())? as i32,
            has_sorted_offsets: shape.has_sorted_offsets(),
            has_no_overlaps: shape.has_no_overlaps(),
            has_no_gaps: shape.has_no_gaps(),
        })))
    }

    fn build(
        _encoding: &ListViewEncoding,
        dtype: &DType,
        len: usize,
        metadata: &ListViewMetadata,
        buffers: &[ByteBuffer],
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
            usize::try_from(metadata.elements_len)?,
        )?;

        // Get offsets with proper type from metadata.
        let offsets = children.get(
            1,
            &DType::Primitive(metadata.offset_ptype(), Nullability::NonNullable),
            len,
        )?;

        // Get sizes with proper type from metadata.
        let sizes = children.get(
            2,
            &DType::Primitive(metadata.size_ptype(), Nullability::NonNullable),
            len,
        )?;

        // Extract shape from metadata.
        let shape = ListViewShape::new(
            metadata.has_sorted_offsets,
            metadata.has_no_overlaps,
            metadata.has_no_gaps,
        );

        ListViewArray::try_new(elements, offsets, sizes, validity, shape)
    }
}
