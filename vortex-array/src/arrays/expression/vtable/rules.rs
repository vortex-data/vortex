// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExpressionArray;
use crate::arrays::ExpressionVTable;
use crate::arrays::StructArray;
use crate::expr::Literal;
use crate::expr::Pack;
use crate::expr::Root;
use crate::expr::root;
use crate::expr::transform::replace;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ReduceRuleSet;
use crate::validity::Validity;

pub(super) const RULES: ReduceRuleSet<ExpressionVTable> = ReduceRuleSet::new(&[
    &ExpressionRootRule,
    &ExpressionCombineRule,
    &ExpressionPackToStructRule,
]);

/// A root expression reduces to just the scope array.
#[derive(Debug)]
struct ExpressionRootRule;
impl ArrayReduceRule<ExpressionVTable> for ExpressionRootRule {
    fn reduce(&self, array: &ExpressionArray) -> VortexResult<Option<ArrayRef>> {
        if array.expression.is::<Root>() {
            return Ok(Some(array.input.clone()));
        }

        if let Some(scalar) = array.expression.as_opt::<Literal>() {
            return Ok(Some(
                ConstantArray::new(scalar.clone(), array.len()).into_array(),
            ));
        }

        Ok(None)
    }
}

/// Combine two ExpressionArrays into a single ExpressionArray.
#[derive(Debug)]
struct ExpressionCombineRule;
impl ArrayReduceRule<ExpressionVTable> for ExpressionCombineRule {
    fn reduce(&self, array: &ExpressionArray) -> VortexResult<Option<ArrayRef>> {
        let Some(child) = array.input.as_opt::<ExpressionVTable>() else {
            return Ok(None);
        };

        // Swap every instance of the parent's root with the child's expression.
        let combined_expression = replace(
            array.expression().clone(),
            &root(),
            child.expression().clone(),
        );

        Ok(Some(
            ExpressionArray::try_new(combined_expression, child.input.clone())?.into_array(),
        ))
    }
}

/// Convert pack expressions into StructArrays.
#[derive(Debug)]
struct ExpressionPackToStructRule;
impl ArrayReduceRule<ExpressionVTable> for ExpressionPackToStructRule {
    fn reduce(&self, array: &ExpressionArray) -> VortexResult<Option<ArrayRef>> {
        let Some(pack) = array.expression().as_opt::<Pack>() else {
            return Ok(None);
        };

        // Transform each child expression into an array.
        let fields = array
            .expression()
            .children()
            .iter()
            .map(|expr| {
                ExpressionArray::try_new(expr.clone(), array.input.clone())
                    .map(IntoArray::into_array)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Some(
            StructArray::try_new(
                pack.names.clone(),
                fields,
                array.len(),
                Validity::from(pack.nullability),
            )?
            .into_array(),
        ))
    }
}
