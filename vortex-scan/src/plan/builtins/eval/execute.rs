// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Applying an expression to each batch.

use vortex_array::IntoArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt as _;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_error::VortexResult;
use vortex_expr::ArrayExprExt;
use vortex_expr::BoundExpression;

use super::Projection;
use crate::execution::operator::Cx;
use crate::execution::operator::Input;
use crate::execution::operator::Operator;
use crate::execution::operator::Step;

/// Applies an expression to each batch.
pub(crate) struct EvalStage {
    expression: BoundExpression,
    projection: Option<Projection>,
}

impl EvalStage {
    pub(crate) fn new(expression: BoundExpression, projection: Option<Projection>) -> Self {
        Self {
            expression,
            projection,
        }
    }
}

impl Operator for EvalStage {
    fn compute(&mut self, input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let Input::Batch(batch) = input else {
            return Ok(Step::End);
        };
        if let Some(projection) = &self.projection
            && let Some(input) = batch.as_opt::<Struct>()
        {
            // Fields arrive as they were cut, such as slices of decoded chunks. Optimizing each
            // reduces those, so the struct needs no optimizing of its own.
            let exec = cx.exec();
            let fields = projection
                .indices()
                .iter()
                .map(|&index| input.unmasked_field(index).optimize_ctx(&*exec))
                .collect::<VortexResult<Vec<_>>>()?;
            debug_assert!(
                projection
                    .fields()
                    .fields()
                    .zip(&fields)
                    .all(|(dtype, field)| &dtype == field.dtype()),
                "projected fields do not match the input struct"
            );
            // SAFETY: the plan checked that each projected field has the dtype of the input field
            // it takes, and every field of a struct holds the struct's rows.
            let output = unsafe {
                StructArray::new_unchecked(
                    fields,
                    projection.fields().clone(),
                    batch.len(),
                    projection.validity().clone(),
                )
            };
            return Ok(Step::Final(output.into_array()));
        }
        Ok(Step::Final(batch.apply_bound(&self.expression)?))
    }
}
