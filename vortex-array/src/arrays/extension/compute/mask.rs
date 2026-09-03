// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::scalar_fn::fns::mask::Mask as MaskExpr;
use crate::scalar_fn::fns::mask::MaskReduce;

impl MaskReduce for Extension {
    fn mask(array: ArrayView<'_, Extension>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_storage =
            MaskExpr::try_new(array.storage_array().clone(), mask.clone())?.into_array();
        Ok(Some(
            ExtensionArray::new(
                array
                    .ext_dtype()
                    .with_nullability(masked_storage.dtype().nullability()),
                masked_storage,
            )
            .into_array(),
        ))
    }
}
