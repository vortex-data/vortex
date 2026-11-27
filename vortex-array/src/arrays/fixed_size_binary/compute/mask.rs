// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeBinary;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use crate::scalar_fn::builtin::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for FixedSizeBinary {
    const VALIDITY_IS_METADATA_ONLY: bool = true;

    fn mask(
        array: ArrayView<'_, FixedSizeBinary>,
        mask: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: the values are unchanged, and masking only removes valid rows.
        Ok(Some(unsafe {
            FixedSizeBinaryArray::new_unchecked_handle(
                array.buffer_handle().clone(),
                array.byte_width(),
                array.len(),
                array.validity()?.and(Validity::Array(mask.clone()))?,
            )
            .into_array()
        }))
    }
}
