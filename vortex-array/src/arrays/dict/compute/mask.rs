// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::scalar_fn::fns::mask::Mask as MaskExpr;
use crate::scalar_fn::fns::mask::MaskReduce;

impl MaskReduce for Dict {
    fn mask(array: ArrayView<'_, Dict>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_codes = MaskExpr::try_new(array.codes().clone(), mask.clone())?.into_array();
        // SAFETY: masking codes doesn't change dict invariants
        Ok(Some(unsafe {
            DictArray::new_unchecked(masked_codes, array.values().clone()).into_array()
        }))
    }
}
