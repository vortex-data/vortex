// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Array;
use crate::IntoArray;
use crate::arrays::expression::ExpressionArray;
use crate::arrays::expression::ExpressionVTable;
use crate::expr::Literal;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ExpressionVTable> for ExpressionVTable {
    fn validity(array: &ExpressionArray) -> VortexResult<Validity> {
        let validity_expression = array
            .expression
            .validity()?
            .optimize_recursive(array.input.dtype())?;

        Ok(match validity_expression.as_opt::<Literal>() {
            None => Validity::Array(
                ExpressionArray {
                    expression: validity_expression,
                    dtype: DType::Bool(Nullability::NonNullable),
                    input: array.input.clone(),
                    stats: Default::default(),
                }
                .into_array(),
            ),
            Some(value) => {
                let is_valid = value
                    .as_bool()
                    .value()
                    .vortex_expect("validity is non-nullable");
                if is_valid {
                    // NOTE(ngates): we know it's not Validity::NonNullable else this vtable
                    //  function would never have been called.
                    Validity::AllValid
                } else {
                    Validity::AllInvalid
                }
            }
        })
    }
}
