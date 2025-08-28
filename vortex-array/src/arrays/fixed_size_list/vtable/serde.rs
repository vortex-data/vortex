// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_ensure};

use super::{FixedSizeListArray, FixedSizeListVTable};
use crate::EmptyMetadata;
use crate::arrays::FixedSizeListEncoding;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;

impl SerdeVTable<FixedSizeListVTable> for FixedSizeListVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &FixedSizeListArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    /// Builds a [`FixedSizeListArray`].
    ///
    /// This method expects 1 or 2 children (a second child indicates a validity array).
    fn build(
        _encoding: &FixedSizeListEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &EmptyMetadata,
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

        let DType::FixedSizeList(element_dtype, list_size, _) = &dtype else {
            vortex_bail!("Expected `DType::FixedSizeList`, got {:?}", dtype);
        };

        let num_elements = len * (*list_size as usize);
        let elements = children.get(0, element_dtype.as_ref(), num_elements)?;

        FixedSizeListArray::try_new(elements, *list_size, validity, len)
    }
}
