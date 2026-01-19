// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::Array;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::expression::ExpressionArray;
use crate::arrays::expression::ExpressionVTable;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ExpressionVTable> for ExpressionVTable {
    fn scalar_at(array: &ExpressionArray, index: usize) -> Scalar {
        let scalar = array.input.scalar_at(index);
        let input = ConstantArray::new(scalar, 1).into_array();
        array
            .expression
            .evaluate(&input)
            .vortex_expect("scalar_at should be fallible")
            .scalar_at(0)
    }
}
