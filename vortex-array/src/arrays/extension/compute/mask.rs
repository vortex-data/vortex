// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::fns::mask::Mask as MaskExpr;
use crate::scalar_fn::fns::mask::MaskReduce;

impl MaskReduce for ExtensionVTable {
    fn mask(array: &ExtensionArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_storage = MaskExpr.try_new_array(
            array.storage().len(),
            EmptyOptions,
            [array.storage().clone(), mask.clone()],
        )?;
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
