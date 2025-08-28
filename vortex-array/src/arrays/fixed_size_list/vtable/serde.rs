// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_ensure};

use super::{FixedSizeListArray, FixedSizeListVTable};
use crate::ProstMetadata;
use crate::arrays::FixedSizeListEncoding;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;

#[derive(Clone, prost::Message)]
pub struct FixedSizeListMetadata {
    #[prost(uint64, tag = "1")]
    len: u64,
    #[prost(uint32, tag = "2")]
    list_size: u32,
}

impl SerdeVTable<FixedSizeListVTable> for FixedSizeListVTable {
    type Metadata = ProstMetadata<FixedSizeListMetadata>;

    fn metadata(array: &FixedSizeListArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(FixedSizeListMetadata {
            len: array.len() as u64,
            list_size: array.list_size(),
        })))
    }

    /// Builds a [`FixedSizeListArray`].
    ///
    /// This method expects 1 or 2 children (a second child indicates a validity array).
    fn build(
        _encoding: &FixedSizeListEncoding,
        dtype: &DType,
        len: usize,
        metadata: &FixedSizeListMetadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FixedSizeListArray> {
        vortex_ensure!(
            buffers.is_empty(),
            "`FixedSizeListVTable::build` got some buffers"
        );

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

        let DType::FixedSizeList(element_dtype, size, _) = &dtype else {
            vortex_bail!("Expected `DType::FixedSizeList`, got {:?}", dtype);
        };
        debug_assert_eq!(
            metadata.list_size, *size,
            "metadata list size is different from dtype"
        );

        let num_elements = metadata.len * metadata.list_size as u64;
        let elements = children.get(0, element_dtype.as_ref(), usize::try_from(num_elements)?)?;

        FixedSizeListArray::try_new(
            elements,
            metadata.list_size,
            validity,
            usize::try_from(metadata.len)?,
        )
    }
}
