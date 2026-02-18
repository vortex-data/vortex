// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::expr::NotReduce;
use crate::scalar::Scalar;

impl NotReduce for ConstantVTable {
    fn invert(array: &ConstantArray) -> VortexResult<Option<ArrayRef>> {
        let value = match array.scalar().as_bool().value() {
            Some(b) => Scalar::bool(!b, array.dtype().nullability()),
            None => Scalar::null(array.dtype().clone()),
        };
        Ok(Some(ConstantArray::new(value, array.len()).into_array()))
    }
}
