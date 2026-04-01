// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::not::NotReduce;

impl NotReduce for Constant {
    fn invert(array: ArrayView<'_, Constant>) -> VortexResult<Option<ArrayRef>> {
        let value = match array.scalar().as_bool().value() {
            Some(b) => Scalar::bool(!b, array.dtype().nullability()),
            None => Scalar::null(array.dtype().clone()),
        };
        Ok(Some(ConstantArray::new(value, array.len()).into_array()))
    }
}
